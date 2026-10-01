import logging
import math

import torch
import torch.nn.functional as F

from open_clip.model import CustomTextCLIP
from open_clip.timm_model import TimmModel

from timm.models.eva import Eva, EvaBlock, AttentionRope
from timm.layers.attention import maybe_add_mask, apply_rot_embed_cat, resolve_self_attn_mask
from timm.layers.attention_pool import AttentionPoolLatent

from typing import Optional, Tuple, List

import timm.models.eva as eva_module
import timm.layers.pos_embed as pos_embed_module

_logger = logging.getLogger(__name__)

# === パッチの適用: モデル生成より前に必ず実行 ===
# OpenVINO変換時に動的な画像サイズに対応するため

def resample_abs_pos_embed(
    posemb: torch.Tensor,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = 'bicubic',
    antialias: bool = True,
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb

# 関数処理を変更
eva_module.resample_abs_pos_embed = resample_abs_pos_embed
pos_embed_module.resample_abs_pos_embed = resample_abs_pos_embed

# 動的な入力に対応するために初期化を変更
original_eva_init = eva_module.Eva.__init__
def patched_eva_init(self, *args, **kwargs):
    kwargs["dynamic_img_size"] = True
    original_eva_init(self, *args, **kwargs)
# __init__ を上書き
eva_module.Eva.__init__ = patched_eva_init

# === パッチの適用終了 ===

def attentionpoollatent_forward(
    self: AttentionPoolLatent, 
    x: torch.Tensor, 
    attn_mask: Optional[torch.Tensor] = None, 
    is_use_dot: bool = False
):
    B, N, C = x.shape

    q_latent = self.latent.expand(B, -1, -1)
    q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn and is_use_dot:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if True:
            attn_map_all = attn[0].mean(dim=0)

            cls_to_patches = attn_map_all[0, 1:]
            patches_to_cls = attn_map_all[1:, 0]

            cls2patch_attn = cls_to_patches

            attn_min, attn_max = cls2patch_attn.min(), cls2patch_attn.max()
            cls2patch_attn = (cls2patch_attn - attn_min) / (attn_max - attn_min + 1e-6)

        attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, self.latent_len, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    x = x + self.mlp(self.norm(x))
    # token
    x = x[:, 0]

    return x, cls2patch_attn

# def custom_attentionpoollatent_forward(
#     self, 
#     x: torch.Tensor, 
#     cluster_labels: torch.Tensor, 
#     k_clusters: int = 12,
#     attn_mask: Optional[torch.Tensor] = None, 
#     normalize: bool = False
# ) -> torch.Tensor:
#     """パッチレベルでクラスタリングした領域のCLSレベルの特徴量を取得 (数値一致・OpenVINOトレース対応版)"""
#     B, N, C = x.shape
    
#     # 勾配切り離し
#     q_latent = self.attn_pool.latent.detach().expand(B, -1, -1)
#     q = self.attn_pool.q(q_latent).reshape(
#         B, self.attn_pool.latent_len, self.attn_pool.num_heads, self.attn_pool.head_dim
#     ).transpose(1, 2)
#     q = self.attn_pool.q_norm(q)

#     # Key / Value を全パッチ分一括計算
#     kv = self.attn_pool.kv(x).reshape(
#         B, N, 2, self.attn_pool.num_heads, self.attn_pool.head_dim
#     ).permute(2, 0, 3, 1, 4)
#     k, v = kv.unbind(0)
#     k = self.attn_pool.k_norm(k)

#     # 0番目の CLS トークンは常に計算対象 (True)
#     cls_mask = torch.ones((1,), dtype=torch.bool, device=x.device)

#     label_token_list: List[torch.Tensor] = []
#     for label in range(k_clusters):
#         # 対象クラスタに属するパッチを判定
#         patch_valid = (cluster_labels == label)
        
#         # SDPA の仕様: True = 計算対象 (Retain), False = 遮断 (-inf)
#         valid_mask = torch.cat([cls_mask, patch_valid], dim=0).view(1, 1, 1, N)

#         if attn_mask is not None:
#             current_attn_mask = valid_mask & attn_mask
#         else:
#             current_attn_mask = valid_mask

#         if self.attn_pool.fused_attn:
#             y = F.scaled_dot_product_attention(q, k, v, attn_mask=current_attn_mask)
#         else:
#             raise RuntimeError("fused_attn is required.")

#         y = y.transpose(1, 2).reshape(B, self.attn_pool.latent_len, C)
#         y = self.attn_pool.proj(y)
#         y = self.attn_pool.proj_drop(y)
#         y = y + self.attn_pool.mlp(self.attn_pool.norm(y))

#         y = y[:, 0]
#         y = self.fc_norm(y)
#         y = self.head_drop(y)
#         cls_token = self.head(y)

#         if normalize:
#             cls_token = F.normalize(cls_token, dim=-1)

#         label_token_list.append(cls_token)

#     return torch.cat(label_token_list, dim=0)


def remove_cls_direction(features: torch.Tensor, alpha=1.0):
    """
    features: [B, P+1, C]
    先頭トークンがCLS。
    戻り値: [B, P+1, C]
    """
    cls = features[:, 0, :]       # [B, C]
    patch = features[:, 1:, :]    # [B, P, C]

    # CLSのノルムは任意でよい
    denom = cls.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # 各patchをCLS方向へ射影する係数
    coeff = (
        (patch * cls[:, None, :]).sum(dim=-1, keepdim=True)
        / denom[:, None, :]
    )

    # CLS方向成分を除去
    cls_component = coeff * cls[:, None, :]
    residual = patch - alpha * cls_component

    return torch.cat([cls[:, None, :], residual], dim=1)


def custom_attentionpoollatent_forward(
    self: Eva,
    x: torch.Tensor,
    cluster_labels: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """パッチレベルでクラスタリングした領域のCLSレベルの特徴量を取得 (K動的対応・ベクトル化版)"""
    B, N, C = x.shape

    # cluster_labels から動的に K (クラスタ数) を取得、あるいは入力形状から取得
    # cluster_labels の形状想定: [N-1] または [B, N-1] など
    # cluster_labels が [N-1] の場合は B 次元を追加
    if cluster_labels.ndim == 1:
        cluster_labels_batch = cluster_labels.unsqueeze(0).expand(B, -1)  # [B, N-1]
    else:
        cluster_labels_batch = cluster_labels

    # 1. Query (q) の計算: B x K x Heads x Latent x HeadDim の形状へブロードキャスト準備
    q_latent = self.attn_pool.latent.detach().expand(B, -1, -1)
    q = (
        self.attn_pool.q(q_latent)
        .reshape(
            B,
            self.attn_pool.latent_len,
            self.attn_pool.num_heads,
            self.attn_pool.head_dim,
        )
        .transpose(1, 2)
    )  # [B, Heads, Latent, HeadDim]
    q = self.attn_pool.q_norm(q)

    # 2. Key / Value (k, v) の全パッチ計算
    kv = (
        self.attn_pool.kv(x)
        .reshape(B, N, 2, self.attn_pool.num_heads, self.attn_pool.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    k, v = kv.unbind(0)  # [B, Heads, N, HeadDim]
    k = self.attn_pool.k_norm(k)

    # k = F.linear(
    #     remove_cls_direction(x),
    #     self.attn_pool.kv.weight[:C, :],
    #     self.attn_pool.kv.bias[:C] if self.attn_pool.kv.bias is not None else None,
    # ).reshape(B, N, self.attn_pool.num_heads, self.attn_pool.head_dim).permute(0, 2, 1, 3)
    # k = self.attn_pool.k_norm(k)

    # v = F.linear(
    #     remove_cls_direction(x),
    #     self.attn_pool.kv.weight[C:, :],
    #     self.attn_pool.kv.bias[C:] if self.attn_pool.kv.bias is not None else None,
    # ).reshape(B, N, self.attn_pool.num_heads, self.attn_pool.head_dim).permute(0, 2, 1, 3)

    # 3. 動的な K (クラスタ数) に向けた Mask の一括作成
    # 0 ~ max(cluster_labels) までのラベルを動的生成 [K, 1]
    max_label = cluster_labels_batch.max()
    labels_arange = torch.arange(
        0, max_label + 1, device=x.device
    ).unsqueeze(1)  # [K, 1]

    # パッチ部分の Mask 作成: [K, B, N-1] -> [B, K, N-1]
    # cluster_labels_batch [B, N-1] と compare
    patch_valid = cluster_labels_batch.unsqueeze(0) == labels_arange.unsqueeze(
        1
    )  # [K, B, N-1]
    patch_valid = patch_valid.permute(1, 0, 2)  # [B, K, N-1]

    # CLS トークン用 Mask (常に True): [B, K, 1]
    K_dynamic = labels_arange.shape[0]
    cls_mask = torch.ones(
        (B, K_dynamic, 1), dtype=torch.bool, device=x.device
    )

    # 全体 Mask: [B, K, N]
    valid_mask = torch.cat([cls_mask, patch_valid], dim=-1)

    # SDPA の attn_mask 形状にアライメント: [B, K, 1, 1, N] または [B * K, Heads, Latent, N] にReshape
    # 処理を単純化するため (B * K) をバッチ次元として統合
    valid_mask = valid_mask.unsqueeze(2).unsqueeze(3)  # [B, K, 1, 1, N]

    if attn_mask is not None:
        # attn_mask がある場合は結合
        current_attn_mask = valid_mask & attn_mask.unsqueeze(1)
    else:
        current_attn_mask = valid_mask

    # SDPA に流し込むため (B * K) に結合
    # q: [B*K, Heads, Latent, HeadDim]
    q_expanded = (
        q.unsqueeze(1).expand(-1, K_dynamic, -1, -1, -1).flatten(0, 1)
    )
    # k, v: [B*K, Heads, N, HeadDim]
    k_expanded = (
        k.unsqueeze(1).expand(-1, K_dynamic, -1, -1, -1).flatten(0, 1)
    )
    v_expanded = (
        v.unsqueeze(1).expand(-1, K_dynamic, -1, -1, -1).flatten(0, 1)
    )

    # mask: [B*K, 1, 1, N]
    mask_expanded = current_attn_mask.flatten(0, 1)

    # 4. SDPA を 1 回でまとめて並列計算
    if self.attn_pool.fused_attn:
        y = F.scaled_dot_product_attention(
            q_expanded, k_expanded, v_expanded, attn_mask=mask_expanded
        )
    else:
        raise RuntimeError("fused_attn is required.")

    # 5. 後処理の一括実行
    # y: [B*K, Heads, Latent, HeadDim] -> [B*K, Latent, C]
    y = y.transpose(1, 2).reshape(
        B * K_dynamic, self.attn_pool.latent_len, C
    )
    y = self.attn_pool.proj(y)
    y = self.attn_pool.proj_drop(y)
    y = y + self.attn_pool.mlp(self.attn_pool.norm(y))

    y = y[:, 0]  # [B*K, C]
    y = self.fc_norm(y)
    y = self.head_drop(y)
    cls_tokens = self.head(y)  # [B*K, OutDim]

    if normalize:
        cls_tokens = F.normalize(cls_tokens, dim=-1)

    return cls_tokens  # [B*K, OutDim] (必要に応じて reshape(B, K, -1) や reshape(-1, OutDim) 可能)


def attention_rope_forward(
    self: AttentionRope,
    x,
    rope: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
):
    """Forward pass for the attention module.

    Args:
        x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
        rope: Rotary position embeddings tensor for position-aware attention
        attn_mask: Optional attention mask to apply during attention computation
        is_causal: If True, use causal (autoregressive) masking

    Returns:
        Tensor of shape (batch_size, sequence_length, dim_out)
    """
    B, N, C = x.shape
    gate = self.gate(x).sigmoid() if self.gate is not None else None

    if self.qkv is not None:
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    else:
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    q, k = self.q_norm(q), self.k_norm(k)

    if rope is not None:
        npt = self.num_prefix_tokens
        half = getattr(self, 'rotate_half', False)
        q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
        k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)

    if self.fused_attn and False:
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=is_causal,
        )
    else:
        q = q * self.scale
        attn_logits = q @ k.transpose(-2, -1)
        attn_bias = resolve_self_attn_mask(N, attn_logits, attn_mask, is_causal)
        attn = maybe_add_mask(attn_logits, attn_bias)
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    if gate is not None:
        x = x * gate
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn_logits


def eva_block_forward(
    self: EvaBlock,
    x: torch.Tensor,
    rope: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    if self.gamma_1 is None:
        x_in = x
        x_norm1 = self.norm1(x_in)
        x_attn, attn_logits = self.attn(x_norm1, rope=rope, attn_mask=attn_mask, is_causal=is_causal)
        x_ap1 = x_in + self.drop_path1(x_attn)
        x_norm2 = self.norm2(x_ap1)
        x_mlp = self.mlp(x_norm2)
        x = x_ap1 + self.drop_path2(x_mlp)
        # x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, is_causal=is_causal))
        # x = x + self.drop_path2(self.mlp(self.norm2(x)))
    else:
        x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, is_causal=is_causal))
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    return x, x_attn, x_mlp, attn_logits


def eva_forward_features(
    self: Eva,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    x = self.patch_embed(x)
    x, rot_pos_embed = self._pos_embed(x)
    x = self.norm_pre(x)

    layer_features = []
    if not self.to_openvino:
        self.forward_hook = {"x_attn": [], "x_mlp": [], "x_attn_logits_head1": [], "x_attn_logits_head2": [], "x_attn_logits_head3": []}

    for i, blk in enumerate(self.blocks):
        if self.to_openvino:
            x = blk(x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)
        else:
            x, x_attn, x_mlp, attn_logits = blk(x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)
        if i + 1 in self.store_layers:
            if self.to_openvino:
                layer_features += [F.normalize(x, dim=-1)]
            else:
                layer_features += [x]
                self.forward_hook["x_attn_logits_head1"] += [attn_logits[:, 0, 0, :].reshape(1, -1, 1)]
                self.forward_hook["x_attn_logits_head2"] += [attn_logits[:, 1, 0, :].reshape(1, -1, 1)]
                self.forward_hook["x_attn_logits_head3"] += [attn_logits[:, 2, 0, :].reshape(1, -1, 1)]
                self.forward_hook["x_attn"] += [x_attn]
                self.forward_hook["x_mlp"] += [x_mlp]

    x = self.norm(x)
    return x, layer_features

def eva_forward_head(self: Eva, x: torch.Tensor) -> torch.Tensor:
    x, cls2patch_attn = self.attn_pool(x)
    x = self.fc_norm(x)
    x = self.head_drop(x)
    cls_token = self.head(x)
    return cls_token, cls2patch_attn

def eva_forward(
    self: Eva,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    patch_token, layer_features = self.forward_features(x, attn_mask=attn_mask, is_causal=is_causal)
    cls_token, cls2patch_attn = self.forward_head(patch_token)
    return cls_token, patch_token, cls2patch_attn, layer_features

def timmmodel_forward(self: TimmModel, x):
    cls_token, patch_token, cls2patch_attn, layer_features = self.trunk(x)
    if self.to_openvino:
        fused_layer_feature = torch.cat(layer_features, dim=2)
        fused_layer_feature = F.normalize(fused_layer_feature, dim=-1)
        return cls_token, patch_token, cls2patch_attn, fused_layer_feature
    else:
        return cls_token, patch_token, cls2patch_attn, layer_features

def encode_image(self: CustomTextCLIP, image, normalize: bool = False):
    cls_token, patch_token, cls2patch_attn, layer_features = self.visual(image)
    if normalize:
        return F.normalize(cls_token, dim=-1), patch_token, cls2patch_attn, layer_features
    else:
        return cls_token, patch_token, cls2patch_attn, layer_features