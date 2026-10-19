#!/usr/bin/env python3
"""Apply P3 RTC track-B patches into a StarVLA checkout.

Usage (on GPU box):
  python3 apply_p3_remote.py --root /root/code/Starvla \\
      --patch-src /tmp/starvla_patch
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path


def backup(path: Path, bak_dir: Path) -> None:
    bak_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, bak_dir / path.name)


def replace_method(text: str, method_name: str, new_method: str, *, decorator: str | None = None) -> str:
    """Replace a class method block starting at optional decorator + def name."""
    if f"STARVLA_RTC_P3" in new_method and "STARVLA_RTC_P3" in text and method_name in text:
        # Heuristic: if marker already near method, skip
        if f"def {method_name}" in text and "STARVLA_RTC_P3" in text:
            return text

    if decorator:
        pattern = rf"(^[ \t]*{re.escape(decorator)}\n[ \t]*def {re.escape(method_name)}\()"
    else:
        pattern = rf"(^[ \t]*def {re.escape(method_name)}\()"
    m = re.search(pattern, text, flags=re.M)
    if not m:
        raise RuntimeError(f"Cannot find method start for {method_name}")
    start = m.start()
    # Determine indent of def line
    def_line_start = text.find("def " + method_name, start)
    line_start = text.rfind("\n", 0, def_line_start) + 1
    indent = re.match(r"[ \t]*", text[line_start:]).group(0)
    # Scan forward to next top-level-of-class def/property at same indent
    pos = m.end()
    while True:
        nxt = re.search(rf"^\{indent}(@|\w)", text[pos:], flags=re.M)
        if not nxt:
            end = len(text)
            break
        abs_i = pos + nxt.start()
        snippet = text[abs_i : abs_i + 80]
        if re.match(rf"{indent}(@\w+|def |class )", snippet):
            # allow continuing decorators belonging to THIS method only if still before body end — we already started at decorator
            if abs_i == start:
                pos = abs_i + 1
                continue
            end = abs_i
            break
        pos = abs_i + 1
    return text[:start] + new_method.rstrip() + "\n\n" + text[end:]


NORM_APPLY = '''
    def apply_actions(self, physical_actions):
        """Physical -> normalized (inverse of ``unapply_actions``).

        # >>> STARVLA_RTC_P3 apply_actions
        """
        import numpy as np
        import torch

        physical_actions = np.asarray(physical_actions, dtype=np.float32)
        assert physical_actions.ndim == 2, (
            f"Expected (T, D); got shape {physical_actions.shape}"
        )

        data = {}
        cursor = 0
        for full_key in self._action_keys:
            dim_k = self._action_key_dims.get(full_key, 1)
            slice_ = physical_actions[..., cursor : cursor + dim_k]
            # transform.apply expects numpy arrays (unlike unapply)
            data[full_key] = np.asarray(slice_, dtype=np.float32)
            cursor += dim_k

        if cursor != physical_actions.shape[-1]:
            raise ValueError(
                f"Sum of per-key dims ({cursor}) != action_dim "
                f"({physical_actions.shape[-1]}). "
                f"action_keys={self._action_keys}, "
                f"action_key_dims={self._action_key_dims}"
            )

        out = self._transform.apply(data)

        parts = []
        for full_key in self._action_keys:
            v = out[full_key]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            parts.append(np.asarray(v))
        return np.concatenate(parts, axis=-1).astype(np.float32)
'''


WRAPPER_METHOD = '''    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        # >>> STARVLA_RTC_P3 wrapper
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    "predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        kwargs = dict(kwargs)
        for drop in ("type", "request_id", "payload", "status", "do_sample", "use_ddim", "num_ddim_steps"):
            kwargs.pop(drop, None)

        rtc_enabled = bool(kwargs.pop("rtc_enabled", False))
        rtc_prefix = kwargs.pop("rtc_prefix", None)
        rtc_prefix_space = str(kwargs.pop("rtc_prefix_space", "physical")).lower()
        inference_delay = int(kwargs.pop("inference_delay", 0) or 0)
        execution_horizon = int(kwargs.pop("execution_horizon", 4) or 4)
        max_guidance_weight = float(kwargs.pop("rtc_max_guidance_weight", 10.0) or 10.0)

        rtc_meta = {
            "applied": False,
            "inference_delay": inference_delay,
            "execution_horizon": execution_horizon,
            "prefix_len": 0,
            "prefix_space_used": rtc_prefix_space,
            "fell_back": False,
            "fallback_reason": "",
        }

        fw_kwargs = dict(kwargs)
        if rtc_enabled and rtc_prefix is not None:
            prefix_arr = np.asarray(rtc_prefix, dtype=np.float32)
            if prefix_arr.ndim == 1:
                prefix_arr = prefix_arr.reshape(1, -1)
            if prefix_arr.size == 0 or prefix_arr.shape[0] == 0:
                rtc_meta["fell_back"] = True
                rtc_meta["fallback_reason"] = "empty_prefix"
                logging.warning("RTC enabled but rtc_prefix empty; falling back to non-RTC.")
            else:
                try:
                    if rtc_prefix_space == "physical":
                        if prefix_arr.shape[-1] == 6:
                            prefix_arr = np.concatenate(
                                [prefix_arr, np.zeros((prefix_arr.shape[0], 1), dtype=np.float32)],
                                axis=-1,
                            )
                        prefix_norm = proc.apply_actions(prefix_arr)
                        rtc_meta["prefix_space_used"] = "normalized"
                    elif rtc_prefix_space == "normalized":
                        prefix_norm = prefix_arr
                    else:
                        raise ValueError(f"Unsupported rtc_prefix_space={rtc_prefix_space!r}")

                    fw_kwargs.update(
                        {
                            "rtc_enabled": True,
                            "prev_chunk_left_over": prefix_norm,
                            "inference_delay": max(0, inference_delay),
                            "execution_horizon": max(1, execution_horizon),
                            "rtc_max_guidance_weight": max_guidance_weight,
                        }
                    )
                    rtc_meta["applied"] = True
                    rtc_meta["prefix_len"] = int(np.asarray(prefix_norm).shape[0])
                except Exception as exc:  # noqa: BLE001
                    rtc_meta["fell_back"] = True
                    rtc_meta["fallback_reason"] = f"prefix_prepare:{exc}"
                    logging.exception("RTC prefix prepare failed; falling back to non-RTC.")
        elif rtc_enabled:
            rtc_meta["fell_back"] = True
            rtc_meta["fallback_reason"] = "missing_prefix"
            logging.warning("RTC enabled but rtc_prefix missing; falling back to non-RTC.")

        try:
            out = self._framework.predict_action(examples=examples, **fw_kwargs)
        except Exception as exc:  # noqa: BLE001
            if rtc_meta["applied"]:
                logging.exception("RTC inference failed; retrying without RTC.")
                rtc_meta["fell_back"] = True
                rtc_meta["fallback_reason"] = f"infer:{exc}"
                rtc_meta["applied"] = False
                for k in (
                    "rtc_enabled",
                    "prev_chunk_left_over",
                    "inference_delay",
                    "execution_horizon",
                    "rtc_max_guidance_weight",
                ):
                    fw_kwargs.pop(k, None)
                out = self._framework.predict_action(examples=examples, **fw_kwargs)
            else:
                raise

        normalized = np.asarray(out["normalized_actions"])
        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        return {"actions": unnorm, "rtc": rtc_meta}
        # <<< STARVLA_RTC_P3 wrapper
'''


HEADER_METHOD = '''    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        init_actions: torch.Tensor = None,
        rtc_enabled: bool = False,
        prev_chunk_left_over=None,
        inference_delay: int = 0,
        execution_horizon: int = 4,
        rtc_max_guidance_weight: float = 10.0,
    ) -> torch.Tensor:
        # >>> STARVLA_RTC_P3 action_head
        compute_dtype = self._resolve_compute_dtype(vl_embs)
        vl_embs = vl_embs.to(dtype=compute_dtype)
        if state is not None:
            state = state.to(dtype=compute_dtype)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        if init_actions is not None:
            actions = init_actions.to(device=device, dtype=compute_dtype)
        else:
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.config.action_dim),
                dtype=compute_dtype,
                device=device,
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = (
            _plain_tensor(self.state_encoder(state)) if state is not None else None
        )

        use_rtc = bool(rtc_enabled) and prev_chunk_left_over is not None
        prefix = None
        if use_rtc:
            from starVLA.model.modules.rtc import (
                RTC_DIM_MASK_DELTA_EEF,
                guide_starvla_velocity,
                prepare_prefix_tensor,
            )

            prefix = prepare_prefix_tensor(
                prev_chunk_left_over,
                batch_size=batch_size,
                chunk_size=int(self.config.action_horizon),
                action_dim=int(self.config.action_dim),
                device=device,
                dtype=compute_dtype,
            )

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = _plain_tensor(self.action_encoder(actions, timesteps_tensor))
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = _plain_tensor(self.position_embedding(pos_ids)).unsqueeze(0)
                action_features = action_features + pos_embs

            ft_ids = torch.arange(
                self.future_tokens.num_embeddings, device=device, dtype=torch.long
            )
            future_tokens = _plain_tensor(self.future_tokens(ft_ids)).unsqueeze(0).expand(
                vl_embs.shape[0], -1, -1
            )
            if state_features is not None:
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            else:
                sa_embs = torch.cat((future_tokens, action_features), dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=_plain_tensor(vl_embs),
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(_plain_tensor(model_output))
            pred_velocity = pred[:, -self.action_horizon :]

            if use_rtc:
                pred_velocity = guide_starvla_velocity(
                    actions,
                    pred_velocity,
                    prefix,
                    inference_delay=int(inference_delay),
                    t_starvla=float(t_cont),
                    execution_horizon=int(execution_horizon),
                    max_guidance_weight=float(rtc_max_guidance_weight),
                    dim_mask=RTC_DIM_MASK_DELTA_EEF,
                )

            actions = actions + dt * pred_velocity
        return actions
        # <<< STARVLA_RTC_P3 action_head
'''


def patch_qwen(path: Path) -> None:
    text = path.read_text()
    if "import contextlib" not in text:
        text = text.replace("import torch\n", "import torch\nimport contextlib\n", 1)

    # Gate inference_mode so RTC autograd works (P5 fix).
    if "STARVLA_RTC_P3 no_inference_mode_when_rtc" not in text:
        old_deco = """    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
"""
        new_deco = """    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        # >>> STARVLA_RTC_P3 no_inference_mode_when_rtc
        # torch.inference_mode blocks RTC autograd; disable it only when RTC is on.
        rtc_on = bool(kwargs.get("rtc_enabled", False)) and kwargs.get("prev_chunk_left_over") is not None
        cm = contextlib.nullcontext() if rtc_on else torch.inference_mode()
        with cm:
            return self._predict_action_impl(examples=examples, **kwargs)

    def _predict_action_impl(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        # <<< STARVLA_RTC_P3 no_inference_mode_when_rtc
        if type(examples) is not list:
            examples = [examples]
"""
        if old_deco not in text:
            raise RuntimeError(f"QwenGR00T @torch.inference_mode predict_action not found in {path}")
        text = text.replace(old_deco, new_deco, 1)
        print(f"[ok] gated inference_mode -> {path}")
    else:
        print(f"[skip] inference_mode gate already in {path}")

    if "STARVLA_RTC_P3 qwen_groot" in text:
        path.write_text(text)
        print(f"[skip] rtc kwargs already in {path}")
        return
    old = """        dit_dtype = self._get_dit_compute_dtype()
        with self._dit_autocast():
            pred_actions = self.action_model.predict_action(
                last_hidden.to(dtype=dit_dtype),
                state_tensor,
                encoder_attention_mask=backbone_attention_mask,
            )
"""
    new = """        dit_dtype = self._get_dit_compute_dtype()
        # >>> STARVLA_RTC_P3 qwen_groot
        rtc_kwargs = {
            "rtc_enabled": bool(kwargs.pop("rtc_enabled", False)),
            "prev_chunk_left_over": kwargs.pop("prev_chunk_left_over", None),
            "inference_delay": int(kwargs.pop("inference_delay", 0) or 0),
            "execution_horizon": int(kwargs.pop("execution_horizon", 4) or 4),
            "rtc_max_guidance_weight": float(kwargs.pop("rtc_max_guidance_weight", 10.0) or 10.0),
        }
        with self._dit_autocast():
            pred_actions = self.action_model.predict_action(
                last_hidden.to(dtype=dit_dtype),
                state_tensor,
                encoder_attention_mask=backbone_attention_mask,
                **rtc_kwargs,
            )
        # <<< STARVLA_RTC_P3 qwen_groot
"""
    if old not in text:
        raise RuntimeError(f"QwenGR00T predict_action call site not found in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"[ok] patched rtc kwargs -> {path}")


def patch_norm(path: Path) -> None:
    text = path.read_text()
    if "def apply_actions(" in text:
        print(f"[skip] apply_actions already in {path}")
        return
    # Insert before next top-level function after unapply_actions, or append near end of class.
    m = re.search(r"def unapply_actions\(self.*?\n(?:.*\n)*?        return np\.concatenate\(parts, axis=-1\)\n", text)
    if not m:
        raise RuntimeError("Could not locate unapply_actions block end")
    insert_at = m.end()
    path.write_text(text[:insert_at] + "\n" + NORM_APPLY + text[insert_at:])
    print(f"[ok] patched apply_actions -> {path}")


def ensure_imports_wrapper(path: Path) -> None:
    text = path.read_text()
    changed = False
    if "import logging" not in text:
        text = "import logging\n" + text
        changed = True
    if "import numpy as np" not in text:
        text = "import numpy as np\n" + text
        changed = True
    if changed:
        path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--patch-src", type=Path, required=True)
    args = ap.parse_args()

    root: Path = args.root
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = root / f".rtc_p3_backup_{stamp}"
    bak.mkdir(parents=True, exist_ok=True)

    src_rtc = args.patch_src / "starVLA" / "model" / "modules" / "rtc"
    dst_rtc = root / "starVLA" / "model" / "modules" / "rtc"
    if dst_rtc.exists():
        shutil.copytree(dst_rtc, bak / "rtc_prev", dirs_exist_ok=True)
    if dst_rtc.exists():
        shutil.rmtree(dst_rtc)
    shutil.copytree(src_rtc, dst_rtc)
    print(f"[ok] installed rtc module -> {dst_rtc}")

    paths = {
        "norm": root / "deployment" / "model_server" / "policy_norm_processor.py",
        "wrapper": root / "deployment" / "model_server" / "policy_wrapper.py",
        "header": root / "starVLA" / "model" / "modules" / "action_model" / "GR00T_ActionHeader.py",
        "qwen": root / "starVLA" / "model" / "framework" / "QwenGR00T.py",
    }
    for p in paths.values():
        backup(p, bak)

    patch_norm(paths["norm"])
    ensure_imports_wrapper(paths["wrapper"])
    w = paths["wrapper"].read_text()
    if "STARVLA_RTC_P3 wrapper" in w:
        print("[skip] wrapper already patched")
    else:
        paths["wrapper"].write_text(replace_method(w, "predict_action", WRAPPER_METHOD))
        print(f"[ok] patched wrapper -> {paths['wrapper']}")

    h = paths["header"].read_text()
    if "STARVLA_RTC_P3 action_head" in h:
        print("[skip] action head already patched")
    else:
        paths["header"].write_text(
            replace_method(h, "predict_action", HEADER_METHOD, decorator="@torch.no_grad()")
        )
        print(f"[ok] patched action head -> {paths['header']}")

    patch_qwen(paths["qwen"])
    print(f"[done] backups at {bak}")


if __name__ == "__main__":
    main()
