import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from functools import partial
from typing import Any

from .mdlm import MDLMTrainer
from dllm.utils.collators import CollatorWrapper
from dllm.utils.data import prepend_bos

# @dataclass
# class AppendEOSBlockWrapper(CollatorWrapper):
#     block_size: int = 32

#     def before(self, features):
#         for ex in features:
#             ids = ex["input_ids"]
#             labs = ex["labels"]

#             assert isinstance(ids, list) and isinstance(labs, list)

#             L = len(ids)
#             target = (L + self.block_size - 1) // self.block_size * self.block_size
#             pad_len = target - L
#             if pad_len > 0:
#                 ex["input_ids"] = ids + [self.tokenizer.eos_token_id] * pad_len
#                 ex["labels"] = labs + [self.tokenizer.eos_token_id] * pad_len
#         return features


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    mask = block_diagonal | offset_block_causal | block_causal

    is_bos_xt_q = (q_idx == 0)
    is_bos_x0_q = (q_idx == n)
    is_any_bos_kv = (kv_idx == 0) | (kv_idx == n)

    # 1) BOS query가 자기 half만 보게
    bos_xt_rule = is_bos_xt_q & (kv_idx < n)          # noisy only (all)
    # BOS_x0는 clean 전체를 봄
    bos_x0_rule = (q_idx == n) & (kv_idx >= n)

    # non-BOS query는 BOS들을 KV로 못 봄 (지름길/누수 차단)
    non_bos_q = (q_idx != 0) & (q_idx != n)
    is_any_bos_kv = (kv_idx == 0) | (kv_idx == n)
    mask = mask & ~(non_bos_q & is_any_bos_kv)

    return mask | bos_xt_rule | bos_x0_rule

def pool_block(model_outputs, position: int | None = None, which: str | None = None):
    """
    Pool a single token embedding from the model outputs.
    For BD3LM, sequences are [noisy | clean] with length 2L. We often want the
    BOS of each half.

    Args:
        position: explicit token position (overrides `which` if not None)
        which: one of {"noisy", "clean", "denoise", "neg"}; unsupported views
               return None so callers can handle missing branches.
    """
    if hasattr(model_outputs, "last_hidden_state"):
        hidden_states = model_outputs.last_hidden_state
    elif hasattr(model_outputs, "hidden_states"):
        hidden_states = model_outputs.hidden_states
    else:
        raise ValueError("Model outputs must include 'last_hidden_state' or 'hidden_states'")

    b, seq_len, _ = hidden_states.shape
    half_len = seq_len // 2

    if position is not None:
        return hidden_states[:, position, :]

    if which == "noisy":
        return hidden_states[:, 0, :]
    if which == "clean":
        return hidden_states[:, half_len, :]
    # Views not present in this model; let caller handle None.
    if which in {"denoise", "neg"}:
        return None

    raise ValueError(f"Unsupported pooling selector: {which}")

class BD3LMContTrainer(MDLMTrainer):
    def __init__(
        self,
        block_size: int = 32,
        gradcache_micro_bsz: int = 8,
        tau: float = 20.0,
        contrastive_mode: str = "gradcache",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.gradcache_micro_bsz = gradcache_micro_bsz
        self.tau = tau
        self.contrastive_mode = contrastive_mode.lower()

        if self.contrastive_mode not in {"vanilla", "gradcache", "empty"}:
            raise ValueError(
                f"Unsupported contrastive_mode '{contrastive_mode}'. "
                "Choose from {'vanilla','gradcache','empty'}."
            )

        print(
            "BD3LMContTrainer initialized with block_size: "
            f"{block_size}, gradcache_micro_bsz: {gradcache_micro_bsz}, "
            f"tau: {tau}, contrastive_mode: {self.contrastive_mode}"
        )

    # ---------- helpers ----------
    def _build_block_attention_mask(self, model, l: int, device: torch.device):
        attn_impl = self.accelerator.unwrap_model(model).config._attn_implementation

        if attn_impl == "sdpa":
            mask = block_diff_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2, device=device)[:, None],
                kv_idx=torch.arange(l * 2, device=device)[None, :],
                block_size=self.block_size,
                n=l,
            )
            # SDPA expects [bsz, heads, q, k] or broadcastable.
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,2l,2l]
            return mask

        elif attn_impl == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask
            mask = create_block_mask(
                partial(block_diff_mask, block_size=self.block_size, n=l),
                B=None, H=None, Q_LEN=l * 2, KV_LEN=l * 2
            )
            return mask

        else:
            raise NotImplementedError(f"Unsupported attn implementation: {attn_impl}")

    def _forward_concat(self, model, concat_input_ids, attention_mask, concat_position_ids, output_hidden_states=True):
        out = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )
        out = self._postprocess_outputs(out)
        return out

    def _contrastive_loss_from_emb(self, q, p, tau: float):
        sim = (q @ p.T) / tau  # [B,B]
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    @torch.no_grad()
    def _collect_embeddings_no_grad(self, model, concat_input_ids, attention_mask, concat_position_ids, micro_bsz: int):
        qs, ps = [], []
        rng_states = []  # <--- 추가
        B = concat_input_ids.size(0)

        for s in range(0, B, micro_bsz):
            e = min(B, s + micro_bsz)

            # forward 직전 RNG state 저장
            rng_states.append(self._capture_rng_state())

            out = self._forward_concat(
                model,
                concat_input_ids[s:e],
                attention_mask,            # broadcastable ok
                concat_position_ids[s:e],
                output_hidden_states=True,
            )
            q = pool_block(out, which="noisy")
            p = pool_block(out, which="clean")
            qs.append(F.normalize(q, dim=-1))
            ps.append(F.normalize(p, dim=-1))

        q_all = torch.cat(qs, dim=0)
        p_all = torch.cat(ps, dim=0)
        return q_all, p_all, rng_states

    def _build_gradcache(self, model, q_local, p_local, tau: float):
        """
        Returns:
          gq_local, gp_local: grads for the local batch embeddings only
          cont_loss_val: detached scalar (global batch contrastive loss)
        """
        acc = self.accelerator
        device = q_local.device
        B_local = q_local.size(0)

        # gather negatives across processes (optional but usually the point)
        if acc.num_processes > 1:
            q_all = acc.gather(q_local)  # [B_global, H]
            p_all = acc.gather(p_local)  # [B_global, H]
            # figure out our slice in the gathered batch
            rank = acc.process_index
            start = rank * B_local
            end = start + B_local
        else:
            q_all, p_all = q_local, p_local
            start, end = 0, B_local

        # enable grad on embeddings only
        q_all = q_all.detach().requires_grad_(True)
        p_all = p_all.detach().requires_grad_(True)

        cont_loss = self._contrastive_loss_from_emb(q_all, p_all, tau=tau)
        cont_loss.backward()  # grads populate on q_all/p_all only

        gq_all = q_all.grad.detach()
        gp_all = p_all.grad.detach()

        gq_local = gq_all[start:end].contiguous()
        gp_local = gp_all[start:end].contiguous()

        return gq_local, gp_local, cont_loss.detach()

    def _capture_rng_state(self):
        state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        torch.set_rng_state(state["cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _prepare_block_batch(self, model: nn.Module, inputs: dict[str, Any]):
        inputs = self._preprocess_inputs(inputs)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask_in = inputs.get("attention_mask", None)

        assert self.processing_class.padding_side == "right"
        b, l = input_ids.shape
        device = input_ids.device

        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
        alpha_t = self.scheduler(t)
        p_mask = 1.0 - alpha_t.unsqueeze(1).expand(b, l)

        masked_indices = (torch.rand((b, l), device=device) < p_mask) & (labels != -100)
        noised_input_ids = torch.where(masked_indices, self.processing_class.mask_token_id, input_ids)

        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)
        base_pos = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        block_attn_mask = self._build_block_attention_mask(model, l=l, device=device)
        loss_weights = self._compute_loss_weights(t=t, inputs=inputs, masked_indices=masked_indices)
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)

        return {
            "inputs": inputs,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask_in": attention_mask_in,
            "b": b,
            "l": l,
            "t": t,
            "masked_indices": masked_indices,
            "concat_input_ids": concat_input_ids,
            "concat_position_ids": concat_position_ids,
            "block_attn_mask": block_attn_mask,
            "loss_weights": loss_weights,
            "effective_lengths": effective_lengths,
        }

    def _training_step_vanilla(self, model: nn.Module, inputs: dict[str, Any]):
        model.train()
        grad_accum_steps = getattr(self.args, "gradient_accumulation_steps", 1)
        scale = 1.0 / max(1, grad_accum_steps)

        batch = self._prepare_block_batch(model, inputs)
        out = self._forward_concat(
            model,
            batch["concat_input_ids"],
            batch["block_attn_mask"],
            batch["concat_position_ids"],
            output_hidden_states=True,
        )

        logits = out.logits[:, : batch["l"]]
        if batch["masked_indices"].any():
            token_loss = F.cross_entropy(
                logits[batch["masked_indices"]],
                batch["input_ids"][batch["masked_indices"]],
                reduction="none",
            )
            token_loss = token_loss * batch["loss_weights"][batch["masked_indices"]]
            ce_loss = torch.sum(token_loss / batch["effective_lengths"][batch["masked_indices"]]) / batch["b"]
        else:
            ce_loss = logits.sum() * 0.0

        q = F.normalize(pool_block(out, which="noisy"), dim=-1)
        p = F.normalize(pool_block(out, which="clean"), dim=-1)
        cont_loss = self._contrastive_loss_from_emb(q, p, tau=self.tau)

        total = (ce_loss + cont_loss) * scale
        self.accelerator.backward(total)

        if self.is_world_process_zero():
            self.log({"dllm_loss": float(ce_loss.detach()), "contrastive_loss": float(cont_loss.detach())})

        return (ce_loss.detach() + cont_loss.detach()) * scale

    def _training_step_empty(self, model: nn.Module, inputs: dict[str, Any]):
        model.train()
        grad_accum_steps = getattr(self.args, "gradient_accumulation_steps", 1)
        scale = 1.0 / max(1, grad_accum_steps)

        batch = self._prepare_block_batch(model, inputs)
        out = self._forward_concat(
            model,
            batch["concat_input_ids"],
            batch["block_attn_mask"],
            batch["concat_position_ids"],
            output_hidden_states=False,
        )

        logits = out.logits[:, : batch["l"]]
        if batch["masked_indices"].any():
            token_loss = F.cross_entropy(
                logits[batch["masked_indices"]],
                batch["input_ids"][batch["masked_indices"]],
                reduction="none",
            )
            token_loss = token_loss * batch["loss_weights"][batch["masked_indices"]]
            ce_loss = torch.sum(token_loss / batch["effective_lengths"][batch["masked_indices"]]) / batch["b"]
        else:
            ce_loss = logits.sum() * 0.0

        self.accelerator.backward(ce_loss * scale)

        if self.is_world_process_zero():
            self.log({"dllm_loss": float(ce_loss.detach()), "contrastive_loss": 0.0})

        return ce_loss.detach() * scale

    # ---------- main custom training_step ----------
    def training_step(self, model: nn.Module, inputs: dict[str, Any], num_items_in_batch=None):
        if self.contrastive_mode == "vanilla":
            return self._training_step_vanilla(model, inputs)
        if self.contrastive_mode == "empty":
            return self._training_step_empty(model, inputs)
        if self.contrastive_mode != "gradcache":
            raise ValueError(f"Unsupported contrastive_mode: {self.contrastive_mode}")

        model.train()

        # HF Trainer typically handles gradient accumulation context outside,
        # but we still scale by grad_accum_steps to match default behavior.
        grad_accum_steps = getattr(self.args, "gradient_accumulation_steps", 1)
        scale = 1.0 / max(1, grad_accum_steps)

        batch = self._prepare_block_batch(model, inputs)

        # ---- (2) contrastive GradCache: collect embeddings w/o grad ----
        micro_bsz = max(1, min(self.gradcache_micro_bsz, batch["b"]))
        q_local, p_local, rng_states = self._collect_embeddings_no_grad(
            model,
            batch["concat_input_ids"],
            batch["block_attn_mask"],
            batch["concat_position_ids"],
            micro_bsz=micro_bsz,
        )

        # ---- (3) build cached grads for local embeddings (global negatives if DDP) ----
        # IMPORTANT: build gradcache with grads enabled (but grads only on q/p tensors)
        # We must avoid mixing with model grads here: we used detached embeddings.
        gq_local, gp_local, cont_loss_val = self._build_gradcache(model, q_local, p_local, tau=self.tau)
        # scale cached grads for grad accumulation
        gq_local = gq_local * scale
        gp_local = gp_local * scale

        # ---- (4) CE loss weights (same as your compute_loss) ----
        loss_weights = batch["loss_weights"]
        effective_lengths = batch["effective_lengths"]

        # ---- (5) microbatch loop: CE backward + cached contrastive backward ----
        total_ce_detached = 0.0
        any_masked = batch["masked_indices"].any().item()
        micro_idx = 0
        for s in range(0, batch["b"], micro_bsz):
            e = min(batch["b"], s + micro_bsz)

            # ✅ 1st pass의 동일 microbatch RNG로 복원
            self._restore_rng_state(rng_states[micro_idx])
            micro_idx += 1

            out = self._forward_concat(
                model,
                batch["concat_input_ids"][s:e],
                batch["block_attn_mask"],
                batch["concat_position_ids"][s:e],
                output_hidden_states=True,
            )

            # ---- CE ----
            logits = out.logits[:, : batch["l"]]
            mb_masked = batch["masked_indices"][s:e]

            ce_loss_mb = 0.0
            if any_masked and mb_masked.any():
                mb_input_ids = batch["input_ids"][s:e]
                mb_loss_w = loss_weights[s:e]
                mb_eff_len = effective_lengths[s:e]

                token_loss = F.cross_entropy(
                    logits[mb_masked], mb_input_ids[mb_masked], reduction="none"
                )
                token_loss = token_loss * mb_loss_w[mb_masked]
                ce_loss_mb = torch.sum(token_loss / mb_eff_len[mb_masked]) / (e - s)
                ce_loss_mb = ce_loss_mb * scale
                total_ce_detached += ce_loss_mb.detach().float()

            # ---- inj (cached grad injection) ----
            q_mb = F.normalize(pool_block(out, which="noisy"), dim=-1)
            p_mb = F.normalize(pool_block(out, which="clean"), dim=-1)

            # if self.is_world_process_zero() and micro_idx == 1:
            #     print("q0 diff:", (q_mb.detach() - q_local[s:e]).abs().mean().item())
            #     print("p0 diff:", (p_mb.detach() - p_local[s:e]).abs().mean().item())

            inj = (q_mb * gq_local[s:e]).sum() + (p_mb * gp_local[s:e]).sum()
            total_mb = ce_loss_mb + inj
            self.accelerator.backward(total_mb)
            # if self.is_world_process_zero() and micro_idx == 1:
            #     print("cont_loss:", float(cont_loss_val))
            #     print("inj:", float(inj.detach()))
            #     print("ce:", float(ce_loss_mb) if torch.is_tensor(ce_loss_mb) else 0.0)
            #     print("||gq||:", float(gq_local[s:e].norm().detach()), "||gp||:", float(gp_local[s:e].norm().detach()))

        # ---- (6) logging / return value ----
        # Return a scalar loss for Trainer to log. We already did backward(s) above.
        # (Use detached values to avoid extra graph retention.)
        # total_ce_detached is already scaled; make it an average-ish scalar for readability.
        ce_val = (total_ce_detached / max(1, (batch["b"] + micro_bsz - 1) // micro_bsz)).detach()
        loss_val = ce_val + cont_loss_val.detach() * scale  # scale for consistency in accumulation

        if self.is_world_process_zero():
            self.log({"dllm_loss": float(ce_val), "contrastive_loss": float(cont_loss_val)})

        return loss_val
