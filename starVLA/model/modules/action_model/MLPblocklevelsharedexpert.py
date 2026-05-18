# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Block-level MoE with shared expert action head.

"""Block-level Mixture-of-Experts action head with an always-on shared expert.

Each MLPResNetBlock is replaced by MoEResNetBlockWithSharedExpert:
    x
    |-- task router (top-k) --> K parallel ExpertBlock  --> task_moe_output
    |-- SharedExpertBlock (always-on)                   --> shared_output
    output = x + task_moe_output + shared_scale * shared_output

The shared expert does NOT participate in top-k routing and is NOT included
in the load-balancing auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
#  Original baseline blocks (kept unchanged for reference / fallback)
# ====================================================================

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


# ====================================================================
#  MoE building blocks
# ====================================================================

class ExpertBlock(nn.Module):
    """A single expert: LayerNorm -> Linear -> ReLU.

    No residual inside — the MoE wrapper handles the residual.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(self.norm(x)))


class MoEResNetBlockWithSharedExpert(nn.Module):
    """Block-level MoE with a dedicated always-on shared expert.

    Forward:
        router_input = router_norm(x)
        gate = top-k softmax(router(router_input))       # task experts only
        task_out = sum_k gate_k * ExpertBlock_k(x)
        shared_out = shared_expert(x)                     # always active
        y = x + task_out + shared_scale * shared_out

    Load-balancing auxiliary loss is computed only over task experts.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        balance_loss_weight: float = 0.01,
        use_shared_expert: bool = True,
        shared_expert_scale: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k if (top_k is not None and top_k < num_experts) else None
        self.balance_loss_weight = balance_loss_weight
        self.use_shared_expert = use_shared_expert
        self.shared_expert_scale = shared_expert_scale

        # Router has its own LayerNorm
        self.router_norm = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Task-specific experts
        self.experts = nn.ModuleList([ExpertBlock(dim) for _ in range(num_experts)])

        # Always-on shared expert
        if use_shared_expert:
            self.shared_expert = ExpertBlock(dim)
        else:
            self.shared_expert = None

        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, dim)
        Returns:
            y: (N, dim)
        """
        N, D = x.shape

        # ---- router (with its own LayerNorm) ----
        router_input = self.router_norm(x)
        router_logits = self.router(router_input)               # (N, K)
        router_probs = F.softmax(router_logits, dim=-1)          # (N, K)

        if self.top_k is not None:
            # sparse top-k routing
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)
            topk_indices = topk_indices.to(torch.long)

            gate = torch.zeros(N, self.num_experts, device=x.device, dtype=x.dtype)
            gate.scatter_(1, topk_indices, topk_weights)  # (N, K)
        else:
            # dense soft-MoE
            gate = router_probs  # (N, K)

        # ---- task expert combination ----
        # Dense expert evaluation avoids per-expert mask.any() CUDA syncs.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        task_out = (expert_outputs * gate.unsqueeze(-1)).sum(dim=1)

        # ---- shared expert (always on) ----
        if self.use_shared_expert and self.shared_expert is not None:
            shared_out = self.shared_expert(x)
            task_out = task_out + self.shared_expert_scale * shared_out

        # ---- load-balancing auxiliary loss (task experts only) ----
        expert_importance = router_probs.mean(dim=0)  # (K,)
        target = torch.full_like(expert_importance, 1.0 / self.num_experts)
        balance_loss = ((expert_importance - target) ** 2).sum()
        self.aux_loss = self.balance_loss_weight * balance_loss

        return x + task_out


class MoEMLPResNetBlockLevelWithSharedExpert(nn.Module):
    """MLP backbone where every internal block is a MoEResNetBlockWithSharedExpert.

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
        use_shared_expert: bool = True,
        shared_expert_scale: float = 0.5,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.moe_blocks = nn.ModuleList([
            MoEResNetBlockWithSharedExpert(
                dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                balance_loss_weight=balance_loss_weight,
                use_shared_expert=use_shared_expert,
                shared_expert_scale=shared_expert_scale,
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


# ====================================================================
#  Unified action head (backward-compatible with original)
# ====================================================================

class L1RegressionActionHead(nn.Module):
    """Action head with optional block-level MoE + shared expert.

    Args:
        input_dim:  dimensionality of VLM hidden states (e.g. 2048)
        hidden_dim: internal MLP width (e.g. 4096)
        action_dim: number of action dimensions (default 7)
        NUM_ACTIONS_CHUNK: number of future steps to predict
        use_moe:     enable block-level MoE (default True)
        num_experts: number of task experts per MoE block (default 4)
        top_k:       top-k sparse routing (default 2); None for dense
        balance_loss_weight: weight of load-balancing loss (default 0.01)
        use_shared_expert:  add always-on shared expert (default True)
        shared_expert_scale: scale factor for shared expert output (default 0.5)
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
        use_shared_expert: bool = True,
        shared_expert_scale: float = 0.5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.use_moe = use_moe

        if use_moe:
            self.model = MoEMLPResNetBlockLevelWithSharedExpert(
                num_blocks=2,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=action_dim,
                num_experts=num_experts,
                top_k=top_k,
                balance_loss_weight=balance_loss_weight,
                use_shared_expert=use_shared_expert,
                shared_expert_scale=shared_expert_scale,
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
        x = self.model(x)
        return x.view(B, T, self.action_dim)

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(actions_hidden_states)

    def get_aux_loss(self) -> torch.Tensor | None:
        """Return the accumulated MoE auxiliary loss, or None if not using MoE."""
        if self.use_moe and hasattr(self.model, "aux_loss"):
            return self.model.aux_loss
        return None


# ====================================================================
#  Factory (backward-compatible signature)
# ====================================================================

def get_action_model(config=None):
    """Build action head from config, with optional MoE + shared expert parameters.

    Reads from ``config.framework.action_model``:
        use_moe             (bool, default True)
        moe_level           (str, default "block")
        num_experts         (int, default 4)
        top_k               (int, default 2)
        balance_loss_weight (float, default 0.01)
        use_shared_expert   (bool, default True)
        shared_expert_scale (float, default 0.5)
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
    use_shared_expert = getattr(action_cfg, "use_shared_expert", True)
    shared_expert_scale = getattr(action_cfg, "shared_expert_scale", 0.5)

    # top_k = 0 or top_k >= num_experts -> dense soft-MoE
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
            use_shared_expert=use_shared_expert,
            shared_expert_scale=shared_expert_scale,
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


# ====================================================================
#  Training usage example
# ====================================================================
# pred_actions = action_model(actions_hidden_states)
# action_loss = F.l1_loss(pred_actions, gt_actions)
# aux_loss = action_model.get_aux_loss()
# loss = action_loss + aux_loss if aux_loss is not None else action_loss
# loss.backward()
