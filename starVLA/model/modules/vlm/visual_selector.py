# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Visual token selector for Qwen3.5-VL — independent of main pipeline.

LightVLAGumbelSelector: a parameter-free module that re-weights vision
tokens by attending them against language tokens, then selects a
permutation via Gumbel-softmax (training) or argmax (inference).

Does NOT modify token count: input [B, Nv, H] → output [B, Nv, H].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightVLAGumbelSelector(nn.Module):
    """Gumbel-softmax visual token selector — zero trainable parameters.

    Forward:
        1. Normalise V and L
        2. Cross-attention V→L to build dynamic query Q
        3. Score matrix S = Qn @ Vnᵀ
        4. Gumbel-softmax selection (train) or argmax (eval)
        5. selected = selection @ V  (same shape as input)

    Args:
        gumbel_tau:      temperature for Gumbel-softmax (default 0.5)
        hard:            straight-through Gumbel (default True)
        cross_attn_tau:  temperature for V→L cross-attention (default 0.1)
    """

    def __init__(
        self,
        gumbel_tau: float = 0.5,
        hard: bool = True,
        cross_attn_tau: float = 0.1,
    ):
        super().__init__()
        self.gumbel_tau = gumbel_tau
        self.hard = hard
        self.cross_attn_tau = cross_attn_tau

    def forward(
        self,
        visual_tokens: torch.Tensor,     # [B, Nv, H]
        language_tokens: torch.Tensor,   # [B, Nl, H]
        return_score_matrix: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            return_score_matrix: if True, include full score_matrix in selector_info.

        Returns:
            selected_visual_tokens:  [B, Nv, H]  same shape / device / dtype as input
            selector_info:           dict with lightweight stats (on CPU, detached)
        """
        B, Nv, H = visual_tokens.shape
        Nl = language_tokens.shape[1]
        orig_dtype = visual_tokens.dtype

        # work in fp32
        V = visual_tokens.float()        # [B, Nv, H]
        L = language_tokens.float()      # [B, Nl, H]

        # normalise
        Vn = F.normalize(V, dim=-1)      # [B, Nv, H]
        Ln = F.normalize(L, dim=-1)      # [B, Nl, H]

        # cross-attention visual → language
        A_v2l = (Vn @ Ln.transpose(-1, -2)) / self.cross_attn_tau   # [B, Nv, Nl]
        A_v2l = F.softmax(A_v2l, dim=-1)                              # [B, Nv, Nl]

        # dynamic query
        Q = A_v2l @ L                      # [B, Nv, H]
        Qn = F.normalize(Q, dim=-1)        # [B, Nv, H]

        # score matrix
        S = Qn @ Vn.transpose(-1, -2)      # [B, Nv, Nv]

        # selection
        if self.training:
            selection = F.gumbel_softmax(
                S, tau=self.gumbel_tau, hard=self.hard, dim=-1
            )  # [B, Nv, Nv]
        else:
            idx = S.argmax(dim=-1)                                   # [B, Nv]
            selection = F.one_hot(idx, num_classes=Nv).to(S.dtype)   # [B, Nv, Nv]

        # apply selection
        selected = selection @ V           # [B, Nv, H]
        selected = selected.to(orig_dtype)

        importance = S.max(dim=1).values          # [B, Nv]
        selected_indices = selection.argmax(dim=-1)  # [B, Nv]

        selector_info: dict = {
            "importance_mean": importance.mean().detach().cpu(),
            "importance_max": importance.max().detach().cpu(),
            "importance_min": importance.min().detach().cpu(),
            "selected_indices": selected_indices.detach().cpu(),
            "selection_shape": tuple(selection.shape),
            "visual_tokens_shape": tuple(visual_tokens.shape),
            "language_tokens_shape": tuple(language_tokens.shape),
        }
        if return_score_matrix:
            selector_info["score_matrix"] = S.detach().cpu()
        return selected, selector_info


if __name__ == "__main__":
    torch.manual_seed(42)

    B, Nv, Nl, H = 2, 256, 32, 4096
    v = torch.randn(B, Nv, H, dtype=torch.bfloat16)
    l = torch.randn(B, Nl, H, dtype=torch.bfloat16)

    selector = LightVLAGumbelSelector(gumbel_tau=0.5, hard=True, cross_attn_tau=0.1)

    # training mode (default: no score_matrix)
    selector.train()
    out_train, info_train = selector(v, l)
    assert out_train.shape == (B, Nv, H), f"train shape {out_train.shape}"
    assert out_train.dtype == v.dtype, f"train dtype {out_train.dtype}"
    assert "score_matrix" not in info_train, "score_matrix should NOT be in default info"
    assert "importance_mean" in info_train
    assert "importance_max" in info_train
    assert "importance_min" in info_train
    assert "selected_indices" in info_train
    print(f"[train]  output shape: {out_train.shape}  dtype: {out_train.dtype}")
    print(f"[train]  importance mean/max/min: {info_train['importance_mean']:.4f} / {info_train['importance_max']:.4f} / {info_train['importance_min']:.4f}")
    print(f"[train]  indices shape: {info_train['selected_indices'].shape}")

    # training mode with score_matrix
    out_train_s, info_train_s = selector(v, l, return_score_matrix=True)
    assert "score_matrix" in info_train_s, "score_matrix should be present when requested"
    print(f"[train+s] score_matrix shape: {info_train_s['score_matrix'].shape}")

    # eval mode (default: no score_matrix)
    selector.eval()
    out_eval, info_eval = selector(v, l)
    assert out_eval.shape == (B, Nv, H), f"eval shape {out_eval.shape}"
    assert out_eval.dtype == v.dtype, f"eval dtype {out_eval.dtype}"
    assert "score_matrix" not in info_eval
    print(f"[eval]   output shape: {out_eval.shape}  dtype: {out_eval.dtype}")

    # gradient check (V needs grad so Gumbel-softmax can backprop)
    selector.train()
    v_grad = v.clone().detach().requires_grad_(True)
    out, _ = selector(v_grad, l)
    loss = out.sum()
    loss.backward()
    assert v_grad.grad is not None, "no gradient through selector"
    print(f"[grad]  backward OK — loss={loss.item():.2f}  v.grad norm={v_grad.grad.norm().item():.4f}")

    # parameters check
    n_params = sum(p.numel() for p in selector.parameters())
    assert n_params == 0, f"unexpected params: {n_params}"
    print(f"[params] trainable params: {n_params}")

    # all tensors in info should be on CPU
    for k, v in info_train.items():
        if torch.is_tensor(v):
            assert v.device.type == "cpu", f"{k} not on CPU: {v.device}"
    print("[cpu]   all info tensors on CPU")

    print("\nAll tests passed.")
