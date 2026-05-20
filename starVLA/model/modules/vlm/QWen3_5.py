# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

from starVLA.model.modules.vlm.visual_selector import LightVLAGumbelSelector

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

        # visual selector (optional, off by default — does not touch forward)
        vsel_cfg = qwenvl_config.get("visual_selector", {})
        self.use_visual_selector = vsel_cfg.get("use_visual_selector", False)
        self._vsel_debug = vsel_cfg.get("visual_selector_debug", False)
        self.last_selector_info = None
        if self.use_visual_selector:
            selector_type = vsel_cfg.get("visual_selector_type", "lightvla_gumbel")
            if selector_type == "lightvla_gumbel":
                self.visual_selector = LightVLAGumbelSelector(
                    cross_attn_tau=vsel_cfg.get("visual_selector_tau", 0.1),
                    gumbel_tau=vsel_cfg.get("visual_selector_gumbel_tau", 0.5),
                    hard=vsel_cfg.get("visual_selector_hard", True),
                )
            else:
                raise ValueError(f"Unknown visual_selector_type: {selector_type}")
        else:
            self.visual_selector = None

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.

        If use_visual_selector=True, a forward hook is temporarily placed on
        ``self.model.model.visual`` to re-weight vision tokens via
        language-conditioned selection before they enter the Transformer.
        The hook is removed immediately after the call so subsequent
        forward passes are unaffected.
        """
        hook_handle = None
        if (
            self.use_visual_selector
            and self.visual_selector is not None
            and kwargs.get("pixel_values") is not None
        ):
            input_ids = kwargs["input_ids"]
            B = input_ids.shape[0]

            # pre-compute language embeddings for the selector
            with torch.autocast("cuda", dtype=torch.bfloat16):
                language_embeds = self.model.model.embed_tokens(input_ids)

            sel_ref = self.visual_selector
            debug_flag = self._vsel_debug

            # per-sample visual-token counts: count IMAGE_TOKEN_INDEX per sample
            sample_patch_counts = (input_ids == IMAGE_TOKEN_INDEX).sum(dim=1).tolist()
            grid_thw = kwargs.get("image_grid_thw", kwargs.get("grid_thw"))

            def _visual_hook(module, input, output):
                """Intercept visual encoder output, run selector, return modified."""
                if isinstance(output, tuple):
                    vision = output[0]
                    rest = output[1:]
                else:
                    vision = output
                    rest = ()

                total_p, H_dim = vision.shape

                # every sample must have the same patch count
                unique_counts = set(sample_patch_counts)
                if len(unique_counts) != 1:
                    if debug_flag:
                        logger.warning(
                            "visual_selector skipped: samples have different patch counts %s "
                            "(grid_thw=%s)",
                            sample_patch_counts,
                            grid_thw.tolist() if grid_thw is not None else "None",
                        )
                    return output

                Nv = sample_patch_counts[0]
                if Nv == 0:
                    return output  # no vision tokens

                # safety: sum must match total vision tokens
                if sum(sample_patch_counts) != total_p:
                    if debug_flag:
                        logger.warning(
                            "visual_selector skipped: patch count mismatch "
                            "(input_ids sum=%d vision output total=%d)",
                            sum(sample_patch_counts), total_p,
                        )
                    return output

                vision_b = vision.view(B, Nv, H_dim)
                selected, info = sel_ref(vision_b, language_embeds)

                # store slim, detached, CPU-safe selector_info
                self.last_selector_info = {
                    k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in info.items()
                    if k not in ("score_matrix", "selection_shape")
                }
                # score_matrix → scalar summaries only (if present)
                sm = info.get("score_matrix")
                if sm is not None:
                    sm_d = sm.detach()
                    self.last_selector_info["score_mean"] = sm_d.mean().cpu()
                    self.last_selector_info["score_max"] = sm_d.max().cpu()
                    self.last_selector_info["score_min"] = sm_d.min().cpu()
                self.last_selector_info["sample_patches_per_image"] = sample_patch_counts

                vision_flat = selected.view(total_p, H_dim)

                if rest:
                    return (vision_flat,) + rest
                return vision_flat

            hook_handle = self.model.model.visual.register_forward_hook(
                _visual_hook
            )

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    **kwargs,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return outputs

    def get_input_embeddings(self, qwen_inputs):
        """
        Extract raw vision and text embeddings before they enter the Transformer.

        Vision images go through the visual encoder and projection, defined by
        Qwen3.5 Model; text tokens go through the embedding lookup.

        Tokens at IMAGE_TOKEN_INDEX positions (248056) in the text are
        placeholders whose embeddings will be replaced by vision features
        inside the model forward.

        Args:
            qwen_inputs: dict from build_qwenvl_inputs(), must contain
                ``pixel_values``, ``grid_thw`` (or ``image_grid_thw``),
                and ``input_ids``.

        Returns:
            vision_embeds: (total_patches, hidden_size) or None if no images
            text_embeds:   (B, seq_len, hidden_size)
        """
        pixel_values = qwen_inputs.get("pixel_values", None)
        grid_thw = qwen_inputs.get("image_grid_thw", qwen_inputs.get("grid_thw", None))
        input_ids = qwen_inputs["input_ids"]

        # vision embedding
        if pixel_values is not None and pixel_values.numel() > 0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vision_outputs = self.model.model.visual(pixel_values, grid_thw=grid_thw)
            if isinstance(vision_outputs, tuple):
                vision_embeds = vision_outputs[0]
            else:
                vision_embeds = vision_outputs
        else:
            vision_embeds = None

        # text embedding (lookup table)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_embeds_all = self.model.model.embed_tokens(input_ids)

        return vision_embeds, text_embeds_all

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
