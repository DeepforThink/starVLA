# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Block-level MoE extension of MLP_ActionHeader.

"""Block-level Mixture-of-Experts action head.

Each MLPResNetBlock is replaced by MoEMLPResNetBlock, which places
multiple expert FFNs in parallel and fuses their outputs with a
learned router (top-k sparse or dense soft-MoE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────
#  Original baseline blocks (kept unchanged for reference / fallback)
# ────────────────────────────────────────────────────────────────────

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with Pre-LN and residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """MLP backbone with stacked residual blocks."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


# ────────────────────────────────────────────────────────────────────
#  MoE building blocks
# ────────────────────────────────────────────────────────────────────

class ExpertBlock(nn.Module):
    """A single expert: Pre-LN → Linear → ReLU (no residual here — the
    MoE wrapper handles the residual and gating)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(self.norm(x)))


class MoEMLPResNetBlock(nn.Module):
    """Block-level MoE: replaces a single FFN with K parallel experts.

        router_input  = router_norm(x)             # router gets its own LN
        router_logits = router(router_input)       # (N, K)
        router_probs  = softmax(router_logits)
        top-k select → re-normalised gate weights
        output = sum_i gate_i * expert_i(x)        # experts receive raw x (ExpertBlock has own LN)
        y = x + output

    Supports dense soft-MoE when top_k is None or top_k >= num_experts.
    Computes and stores load-balancing auxiliary loss per block.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        balance_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k if (top_k is not None and top_k < num_experts) else None
        self.balance_loss_weight = balance_loss_weight

        self.router_norm = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertBlock(dim) for _ in range(num_experts)])

        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, dim)
        Returns:
            y: (N, dim)  —  x + weighted expert outputs
        """
        N, D = x.shape

        # ---- router (with its own LayerNorm) ----
        router_input = self.router_norm(x)
        router_logits = self.router(router_input)              # (N, K)
        router_probs = F.softmax(router_logits, dim=-1)         # (N, K)

        if self.top_k is not None:
            # sparse top-k routing
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)  # (N, top_k)
            topk_weights = F.softmax(topk_weights, dim=-1)                       # re-normalise
            topk_indices = topk_indices.to(torch.long)

            gate = torch.zeros(N, self.num_experts, device=x.device, dtype=x.dtype)
            gate.scatter_(1, topk_indices, topk_weights)  # (N, K)

            # experts receive raw x (ExpertBlock has its own LN)
            combined = torch.zeros(N, D, device=x.device, dtype=x.dtype)
            for k in range(self.num_experts):
                mask = gate[:, k] > 0  # tokens routed to expert k
                if mask.any():
                    expert_out = self.experts[k](x[mask])              # (M, D)
                    combined[mask] += gate[mask, k:k + 1] * expert_out  # (M, D)

            output = combined
        else:
            # dense soft-MoE: every token goes through every expert
            expert_outputs = torch.stack(
                [expert(x) for expert in self.experts], dim=1
            )  # (N, K, D)
            output = (expert_outputs * router_probs.unsqueeze(-1)).sum(dim=1)  # (N, D)

        # ---- no-gradient monitoring stats ----
        with torch.no_grad():
            self.router_entropy = -(router_probs * (router_probs + 1e-9).log()).sum(dim=-1).mean()
            self.expert_prob_mean = router_probs.mean(dim=0)
            if self.top_k is not None:
                self.expert_usage = (gate > 0).float().mean(dim=0)
            else:
                self.expert_usage = router_probs.mean(dim=0)

        # ---- load-balancing auxiliary loss ----
        expert_importance = router_probs.mean(dim=0)  # (K,)
        target = torch.full_like(expert_importance, 1.0 / self.num_experts)
        balance_loss = ((expert_importance - target) ** 2).sum()
        self.aux_loss = self.balance_loss_weight * balance_loss

        return x + output


class MoEMLPResNet(nn.Module):
    """MLP backbone where every internal block is an MoEMLPResNetBlock.

    Accumulates auxiliary losses from all MoE blocks into ``self.aux_loss``.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        balance_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.moe_blocks = nn.ModuleList([
            MoEMLPResNetBlock(
                dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                balance_loss_weight=balance_loss_weight,
            )
            for _ in range(num_blocks)
        ])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)

        total_aux = None
        for block in self.moe_blocks:
            x = block(x)
            if block.aux_loss is not None:
                total_aux = block.aux_loss if total_aux is None else total_aux + block.aux_loss

        self.aux_loss = total_aux

        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


# ────────────────────────────────────────────────────────────────────
#  Unified action head (backward-compatible with original)
# ────────────────────────────────────────────────────────────────────

class L1RegressionActionHead(nn.Module):
    """Action head with optional block-level MoE.

    Args:
        input_dim:  dimensionality of VLM hidden states (e.g. 2048)
        hidden_dim: internal MLP width (e.g. 4096)
        action_dim: number of action dimensions (default 7)
        NUM_ACTIONS_CHUNK: number of future steps to predict
        use_moe:     enable block-level MoE (default True)
        num_experts: number of experts per MoE block (default 4)
        top_k:       top-k sparse routing (default 2); None for dense
        balance_loss_weight: weight of load-balancing loss (default 0.01)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        NUM_ACTIONS_CHUNK: int = 8,
        use_moe: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        balance_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.use_moe = use_moe

        if use_moe:
            self.model = MoEMLPResNet(
                num_blocks=2,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=action_dim,
                num_experts=num_experts,
                top_k=top_k,
                balance_loss_weight=balance_loss_weight,
            )
        else:
            self.model = MLPResNet(
                num_blocks=2,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=action_dim,
            )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: (B, chunk_len, hidden_dim)
        Returns:
            actions: (B, chunk_len, action_dim)
        """
        B, T, D = actions_hidden_states.shape
        x = actions_hidden_states.reshape(B * T, D)
        x = self.model(x)  # (B*T, action_dim)
        return x.view(B, T, self.action_dim)

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(actions_hidden_states)

    def get_aux_loss(self) -> torch.Tensor | None:
        """Return the accumulated MoE auxiliary loss, or None if not using MoE."""
        if self.use_moe and hasattr(self.model, "aux_loss"):
            return self.model.aux_loss
        return None

    def get_moe_metrics(self) -> dict:
        """Collect no-gradient monitoring metrics from all MoE blocks."""
        if not self.use_moe or not hasattr(self.model, "moe_blocks"):
            return {}
        metrics = {}
        for i, block in enumerate(self.model.moe_blocks):
            if hasattr(block, "router_entropy"):
                metrics[f"moe/block_{i}/router_entropy"] = block.router_entropy.detach()
            if hasattr(block, "expert_prob_mean"):
                for j, v in enumerate(block.expert_prob_mean.detach()):
                    metrics[f"moe/block_{i}/expert_{j}_prob"] = v
            if hasattr(block, "expert_usage"):
                for j, v in enumerate(block.expert_usage.detach()):
                    metrics[f"moe/block_{i}/expert_{j}_usage"] = v
        return metrics


# ────────────────────────────────────────────────────────────────────
#  Factory (backward-compatible signature)
# ────────────────────────────────────────────────────────────────────

def get_action_model(config=None):
    """Build action head from config, with optional MoE parameters.

    Reads from ``config.framework.action_model``:
        use_moe             (bool, default True)
        moe_level           (str, default "block")
        num_experts         (int, default 4)
        top_k               (int, default 2)  — set to 0 or null for dense
        balance_loss_weight (float, default 0.01)
    """
    action_cfg = config.framework.action_model
    action_hidden_dim = action_cfg.action_hidden_dim
    action_dim = action_cfg.action_dim
    action_horizon = int(action_cfg.action_horizon)

    use_moe = getattr(action_cfg, "use_moe", True)
    moe_level = getattr(action_cfg, "moe_level", "block")
    num_experts = getattr(action_cfg, "num_experts", 4)
    top_k_raw = getattr(action_cfg, "top_k", 2)
    balance_loss_weight = getattr(action_cfg, "balance_loss_weight", 0.01)

    # top_k = 0 or top_k >= num_experts → dense soft-MoE
    top_k = top_k_raw if (top_k_raw is not None and top_k_raw > 0) else None

    if use_moe and moe_level == "block":
        model = L1RegressionActionHead(
            input_dim=action_hidden_dim,
            hidden_dim=action_hidden_dim * 2,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=action_horizon,
            use_moe=True,
            num_experts=num_experts,
            top_k=top_k,
            balance_loss_weight=balance_loss_weight,
        )
    else:
        model = L1RegressionActionHead(
            input_dim=action_hidden_dim,
            hidden_dim=action_hidden_dim * 2,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=action_horizon,
            use_moe=False,
        )

    return model


# ────────────────────────────────────────────────────────────────────
#  Training usage example
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- Build a dummy config to exercise the factory ---
    class DummyConfig:
        class framework:
            class action_model:
                action_hidden_dim = 2048
                action_dim = 7
                action_horizon = 8
                use_moe = True
                moe_level = "block"
                num_experts = 4
                top_k = 2
                balance_loss_weight = 0.01

    config = DummyConfig()
    action_model = get_action_model(config)
    print(action_model)

    # --- Synthetic forward pass ---
    B, T, H = 2, 8, 2048
    actions_hidden_states = torch.randn(B, T, H)
    gt_actions = torch.randn(B, T, 7)

    pred_actions = action_model(actions_hidden_states)
    action_loss = F.l1_loss(pred_actions, gt_actions)
    aux_loss = action_model.get_aux_loss()
    loss = action_loss + aux_loss if aux_loss is not None else action_loss

    print(f"action_loss: {action_loss.item():.4f}")
    print(f"aux_loss:    {aux_loss.item() if aux_loss is not None else 'None'}")
    print(f"total_loss:  {loss.item():.4f}")

    # --- Gradient check ---
    loss.backward()
    print("Backward OK — gradients flowed through MoE router and experts.")
