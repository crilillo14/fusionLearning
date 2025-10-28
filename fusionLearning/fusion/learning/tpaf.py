# # ! gpt 5... rescue whatever

# import math

# from dataclasses import dataclass
# from typing import List, Tuple, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -------------------------
# # Config
# # -------------------------

# @dataclass
# class FusionConfig:
#     patch_size: int = 16                         # non-overlapping patch side (s x s)
#     min_required_height: int = 16                # padding policy will ensure divisibility
#     min_required_width: int = 16

#     # Feature toggles for patch descriptors
#     use_patch_entropy: bool = True
#     use_confidence_gap: bool = True
#     use_boundary_ratio: bool = False             # can be enabled; cheap approx provided

#     # Attention sizes
#     model_axis_mlp_hidden: int = 32
#     class_axis_mlp_hidden: int = 64
#     model_class_lowrank_rank: int = 8            # r in the writeup

#     # Temperatures
#     learn_per_model_temperature: bool = True
#     initial_temperature_value: float = 1.0

#     # Regularization / numerics
#     epsilon_for_numerics: float = 1e-8

#     # Optional: clamp logits before softmax for entropy calc
#     entropy_logits_clamp: float = 10.0


# # -------------------------
# # Small utility modules
# # -------------------------

# class SimpleMLP(nn.Module):
#     """One-hidden-layer MLP with GELU, small and readable."""
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#         # xavier init for stability
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.fc2(self.drop(self.act(self.fc1(x))))


# # -------------------------
# # Core fusion module
# # -------------------------

# class TinyPatchAttentionFusion(nn.Module):
#     """
#     Patch-wise, class-aware attention to fuse N model logits (B, C, H, W) -> fused logits (B, C, H, W).
#     Keeps the attention constant within each non-overlapping patch for stability and efficiency.

#     Inputs: list of N tensors, each of shape (B, C, H, W), *logits* (not softmax) for the same image.
#     Output: fused logits (B, C, H, W).

#     Notes:
#       - Complexity is linear in N and pixels.
#       - No heavy token mixing; only per-patch descriptors + tiny MLPs + low-rank interactions.
#     """
#     def __init__(self, number_of_models: int, number_of_classes: int, config: FusionConfig):
#         super().__init__()
#         self.number_of_models = number_of_models
#         self.number_of_classes = number_of_classes
#         self.cfg = config

#         # ---- Per-model temperature (optional) ----
#         if self.cfg.learn_per_model_temperature:
#             initial_log_temp = math.log(self.cfg.initial_temperature_value + self.cfg.epsilon_for_numerics)
#             self.per_model_log_temperature = nn.Parameter(
#                 torch.full((number_of_models,), initial_log_temp, dtype=torch.float32)
#             )
#         else:
#             self.register_buffer("per_model_log_temperature",
#                                  torch.log(torch.tensor([self.cfg.initial_temperature_value] * number_of_models)))

#         # ---- Patch feature dimensionality ----
#         # Per-class mean logits (C) are summarized downstream; the patch descriptor is small:
#         #   - entropy (1) if enabled
#         #   - confidence gap top1-top2 (1) if enabled
#         #   - boundary ratio (1) if enabled
#         descriptor_dim = 0
#         if self.cfg.use_patch_entropy:
#             descriptor_dim += 1
#         if self.cfg.use_confidence_gap:
#             descriptor_dim += 1
#         if self.cfg.use_boundary_ratio:
#             descriptor_dim += 1

#         self.patch_descriptor_dim = descriptor_dim if descriptor_dim > 0 else 1  # avoid 0-dim

#         # ---- Model-axis reliability MLP (class-agnostic) ----
#         # Input: patch descriptor for (model, patch)
#         # Output: scalar score, later normalized across models via softmax
#         self.model_axis_mlp = SimpleMLP(
#             input_dim=self.patch_descriptor_dim,
#             hidden_dim=self.cfg.model_axis_mlp_hidden,
#             output_dim=1
#         )

#         # ---- Class-axis reweighting MLP (class-aware) ----
#         # Input: per-patch average of per-model mean logits across models -> vector in R^C
#         # Output: gamma_kc in R^C
#         self.class_axis_mlp = SimpleMLP(
#             input_dim=number_of_classes,
#             hidden_dim=self.cfg.class_axis_mlp_hidden,
#             output_dim=number_of_classes
#         )

#         # ---- Low-rank model×class interaction ----
#         r = self.cfg.model_class_lowrank_rank
#         self.model_embedding = nn.Parameter(torch.randn(number_of_models, r) * 0.02)  # u_n
#         self.class_left_factors = nn.Parameter(torch.randn(number_of_classes, r) * 0.02)  # a_c
#         self.class_right_factors = nn.Parameter(torch.randn(number_of_classes, r) * 0.02) # b_c
#         # context projection: maps per-patch class context (R^C) -> R^r
#         self.class_context_to_rank = nn.Linear(number_of_classes, r)
#         nn.init.xavier_uniform_(self.class_context_to_rank.weight)
#         nn.init.zeros_(self.class_context_to_rank.bias)

#     # ---------- public forward ----------

#     def forward(self, list_of_model_logits: List[torch.Tensor]) -> torch.Tensor:
#         """
#         Args:
#             list_of_model_logits: list length N, each tensor (B, C, H, W), same spatial dims.

#         Returns:
#             fused_logits: (B, C, H, W)
#         """
#         self._validate_and_infer_shapes(list_of_model_logits)

#         # Stack: (B, N, C, H, W)
#         batched_model_logits = torch.stack(list_of_model_logits, dim=1)

#         # Optionally pad so H and W are divisible by patch_size (record unpad)
#         batched_model_logits, pad_info = self._maybe_pad_for_patches(batched_model_logits)

#         B, N, C, Hp, Wp = batched_model_logits.shape
#         s = self.cfg.patch_size
#         P = (Hp // s) * (Wp // s)  # number of non-overlapping patches

#         # Temperature scaling per model (broadcast over B,C,H,W)
#         temperatures = torch.exp(self.per_model_log_temperature).clamp_min(self.cfg.epsilon_for_numerics)  # (N,)
#         scaled_logits = batched_model_logits / temperatures.view(1, N, 1, 1, 1)

#         # Per-(B,N,patch,C) mean logits inside patch (mu_{n,k,c})
#         per_patch_mean_logits = self._compute_per_patch_mean_logits(scaled_logits, patch_size=s)  # (B,N,P,C)

#         # Per-(B,N,patch) patch descriptor z_{n,k}
#         patch_descriptors = self._compute_patch_descriptors(per_patch_mean_logits, scaled_logits, s)  # (B,N,P,Dz)

#         # Model-axis reliability scores -> pi_{n,k} via softmax along model dimension
#         model_axis_scores = self.model_axis_mlp(patch_descriptors)  # (B,N,P,1)
#         model_axis_scores = model_axis_scores.squeeze(-1)           # (B,N,P)
#         model_axis_weights = F.softmax(model_axis_scores, dim=1)    # (B,N,P)

#         # Patch class context: average mean logits over models -> (B,P,C)
#         mean_logits_across_models = per_patch_mean_logits.mean(dim=1)            # (B,P,C)
#         class_axis_adjustment = self.class_axis_mlp(mean_logits_across_models)   # (B,P,C)

#         # Low-rank model×class term delta_{n,k,c}
#         #   A_nc = u_n @ a_c^T -> (N,C)
#         #   v_k = context_to_rank(mu_bar_k) -> (B,P,r); B_kc = v_k @ b_c^T -> (B,P,C)
#         A_nc = self.model_embedding @ self.class_left_factors.t()  # (N, C)
#         v_k = self.class_context_to_rank(mean_logits_across_models)  # (B, P, r)
#         B_kc = torch.einsum('bpr,cr->bpc', v_k, self.class_right_factors)  # (B,P,C)
#         # Broadcast multiply to get delta (B,N,P,C)
#         lowrank_delta = A_nc.view(1, N, 1, C) * B_kc.view(B, 1, P, C)     # (B,N,P,C)

#         # Combine terms -> alpha_{n,k,c} via softmax over models
#         # log pi_{n,k} + gamma_{k,c} + delta_{n,k,c}
#         combined_scores = (
#             torch.log(model_axis_weights + self.cfg.epsilon_for_numerics).unsqueeze(-1) +  # (B,N,P,1)
#             class_axis_adjustment.unsqueeze(1) +                                           # (B,1,P,C)
#             lowrank_delta                                                                   # (B,N,P,C)
#         )  # -> (B,N,P,C)

#         per_model_class_patch_weights = F.softmax(combined_scores, dim=1)  # (B,N,P,C)

#         # Broadcast patch weights back to per-pixel maps and fuse logits
#         fused_logits = self._apply_patch_weights_and_fuse(
#             per_model_class_patch_weights=per_model_class_patch_weights,  # (B,N,P,C)
#             scaled_logits=scaled_logits,                                  # (B,N,C,Hp,Wp)
#             patch_size=s
#         )  # (B,C,Hp,Wp)

#         # Unpad if padding was applied
#         fused_logits = self._maybe_unpad(fused_logits, pad_info)  # (B,C,H,W)

#         return fused_logits

#     # ---------- shape / padding helpers ----------

#     def _validate_and_infer_shapes(self, list_of_model_logits: List[torch.Tensor]) -> None:
#         assert len(list_of_model_logits) == self.number_of_models, \
#             f"Expected {self.number_of_models} model outputs, got {len(list_of_model_logits)}."
#         B, C, H, W = list_of_model_logits[0].shape
#         for idx, x in enumerate(list_of_model_logits):
#             assert x.ndim == 4, f"Model {idx} logits must be (B,C,H,W)"
#             assert x.shape == (B, C, H, W), f"All model outputs must share shape; mismatch at model {idx}"
#         assert C == self.number_of_classes, f"num_classes mismatch: cfg={self.number_of_classes}, input={C}"
#         assert H >= self.cfg.min_required_height and W >= self.cfg.min_required_width, \
#             f"Input too small for patching: {(H,W)}"

#     def _maybe_pad_for_patches(self, stacked_logits: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
#         """Pad right and bottom so H,W are divisible by patch_size. Returns (padded, pad_info=(pad_l,pad_r,pad_t,pad_b))."""
#         B, N, C, H, W = stacked_logits.shape
#         s = self.cfg.patch_size
#         pad_h = (s - (H % s)) % s
#         pad_w = (s - (W % s)) % s
#         if pad_h == 0 and pad_w == 0:
#             return stacked_logits, (0, 0, 0, 0)
#         # F.pad uses (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
#         padded = F.pad(stacked_logits, (0, pad_w, 0, pad_h))
#         return padded, (0, pad_w, 0, pad_h)

#     def _maybe_unpad(self, tensor_hw: torch.Tensor, pad_info: Tuple[int,int,int,int]) -> torch.Tensor:
#         pad_l, pad_r, pad_t, pad_b = pad_info
#         if (pad_l | pad_r | pad_t | pad_b) == 0:
#             return tensor_hw
#         _, _, H, W = tensor_hw.shape
#         return tensor_hw[..., pad_t: H - pad_b, pad_l: W - pad_r]

#     # ---------- patch stats ----------

#     def _compute_per_patch_mean_logits(self, scaled_logits: torch.Tensor, patch_size: int) -> torch.Tensor:
#         """
#         Compute mean logits per patch for each (B, N, C). Output shape: (B, N, P, C).
#         """
#         B, N, C, H, W = scaled_logits.shape
#         s = patch_size

#         # reshape into patches: (B, N, C, H//s, s, W//s, s) -> average over the s*s dims
#         new_h, new_w = H // s, W // s
#         x = scaled_logits.view(B, N, C, new_h, s, new_w, s)
#         per_patch_mean = x.mean(dim=(4, 6))  # average over the patch pixels
#         # (B, N, C, new_h, new_w) -> (B, N, P, C)
#         per_patch_mean = per_patch_mean.permute(0, 1, 3, 4, 2).contiguous().view(B, N, new_h * new_w, C)
#         return per_patch_mean  # (B,N,P,C)

#     def _compute_patch_descriptors(
#         self,
#         per_patch_mean_logits: torch.Tensor,   # (B,N,P,C)
#         scaled_logits: torch.Tensor,           # (B,N,C,H,W)
#         patch_size: int
#     ) -> torch.Tensor:
#         """
#         Build a tiny descriptor per (B,N,P) using logits only.
#         Returns: (B, N, P, Dz)
#         """
#         B, N, P, C = per_patch_mean_logits.shape
#         descriptors = []

#         if self.cfg.use_patch_entropy:
#             # entropy from softmax over mean logits
#             mean_logits_clamped = per_patch_mean_logits.clamp(
#                 min=-self.cfg.entropy_logits_clamp, max=self.cfg.entropy_logits_clamp
#             )
#             probabilities = F.softmax(mean_logits_clamped, dim=-1)  # (B,N,P,C)
#             entropy = -(probabilities * (probabilities.clamp_min(self.cfg.epsilon_for_numerics)).log()).sum(dim=-1)
#             descriptors.append(entropy.unsqueeze(-1))  # (B,N,P,1)

#         if self.cfg.use_confidence_gap:
#             top2_vals, _ = torch.topk(per_patch_mean_logits, k=min(2, C), dim=-1)  # (B,N,P,2)
#             if C >= 2:
#                 gap = top2_vals[..., 0] - top2_vals[..., 1]
#             else:
#                 gap = top2_vals[..., 0]  # degenerate case C==1
#             descriptors.append(gap.unsqueeze(-1))  # (B,N,P,1)

#         if self.cfg.use_boundary_ratio:
#             boundary_ratio = self._approx_boundary_ratio_per_patch(scaled_logits, patch_size)  # (B,N,P)
#             descriptors.append(boundary_ratio.unsqueeze(-1))  # (B,N,P,1)

#         if len(descriptors) == 0:
#             # Fallback: zeros descriptor so shapes work out
#             descriptors = [torch.zeros(B, N, P, 1, device=per_patch_mean_logits.device, dtype=per_patch_mean_logits.dtype)]

#         return torch.cat(descriptors, dim=-1)  # (B,N,P,Dz)

#     def _approx_boundary_ratio_per_patch(self, scaled_logits: torch.Tensor, patch_size: int) -> torch.Tensor:
#         """
#         Cheap boundary score: fraction of pixels in the patch whose argmax label differs from any 4-neighbor.
#         Returns (B, N, P).
#         """
#         B, N, C, H, W = scaled_logits.shape
#         with torch.no_grad():
#             hard_labels = scaled_logits.argmax(dim=2)  # (B,N,H,W)

#             # compare with right and down neighbors
#             right_diff = (hard_labels[..., :, 1:] != hard_labels[..., :, :-1]).float()
#             down_diff = (hard_labels[..., 1:, :] != hard_labels[..., :-1, :]).float()

#             # pad to original size
#             right_diff = F.pad(right_diff, (0, 1, 0, 0))  # pad rightmost column
#             down_diff = F.pad(down_diff, (0, 0, 0, 1))   # pad bottom row

#             # boundary if differs in either direction
#             boundary_map = torch.clamp(right_diff + down_diff, max=1.0)  # (B,N,H,W)

#             # average over patch pixels -> (B,N,P)
#             s = patch_size
#             new_h, new_w = H // s, W // s
#             x = boundary_map.view(B, N, new_h, s, new_w, s)
#             per_patch_boundary = x.mean(dim=(3, 5))           # (B,N,new_h,new_w)
#             per_patch_boundary = per_patch_boundary.view(B, N, new_h * new_w)
#             return per_patch_boundary

#     # ---------- fuse back to pixels ----------

#     def _apply_patch_weights_and_fuse(
#         self,
#         per_model_class_patch_weights: torch.Tensor,  # (B,N,P,C)
#         scaled_logits: torch.Tensor,                  # (B,N,C,Hp,Wp)
#         patch_size: int
#     ) -> torch.Tensor:
#         """
#         Broadcast per-patch weights to every pixel in the patch, then weighted-sum logits over models.
#         """
#         B, N, C, Hp, Wp = scaled_logits.shape
#         s = patch_size
#         new_h, new_w = Hp // s, Wp // s
#         P = new_h * new_w

#         # Expand weights to per-pixel within patch:
#         # (B,N,P,C) -> (B,N,new_h,new_w,C) -> (B,N,C,new_h,s,new_w,s) -> tile within patch
#         weights = per_model_class_patch_weights.view(B, N, new_h, new_w, C).permute(0,1,4,2,3)  # (B,N,C,new_h,new_w)
#         weights = weights.unsqueeze(-1).unsqueeze(-1)                                         
