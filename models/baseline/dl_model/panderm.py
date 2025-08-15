#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch.nn.modules.batchnorm import _NormBase

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt


# In[ ]:


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class LP_BatchNorm(_NormBase):
    """ A variant used in linear probing.
    To freeze parameters (normalization operator specifically), model set to eval mode during linear probing.
    According to paper, an extra BN is used on the top of encoder to calibrate the feature magnitudes.
    In addition to self.training, we set another flag in this implement to control BN's behavior to train in eval mode.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(LP_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input, is_train):
        """
        We use is_train instead of self.training.
        """
        self._check_input_dim(input)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # if self.training and self.track_running_stats:
        if is_train and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if is_train:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not is_train or self.track_running_stats else None,
            self.running_var if not is_train or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x) # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)   
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()

        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.cross_attn =  CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x_q = self.norm_q(x_q + pos_q)
        x_k = self.norm_k(x_kv + pos_k)
        x_v = self.norm_v(x_kv)

        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, init_values=0.1, use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, use_mean_pooling=False, init_scale=0.001, lin_probe=True, 
                 linear_type='standard', args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = self.build_2d_sincos_position_embedding(embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)

        self.lin_probe = lin_probe
        self.linear_type = linear_type

        if lin_probe:
            if self.linear_type == 'standard':
                self.fc_norm = None
            elif self.linear_type == 'attentive':
                self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.attentive_blocks = nn.ModuleList([
                    AttentiveBlock(
                        dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer) 
                        for i in range(1)])
                self.fc_norm = LP_BatchNorm(embed_dim, affine=False)
        else:
            if use_mean_pooling:
                self.fc_norm = norm_layer(embed_dim)
            else:
                self.fc_norm = None

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000.):
        h, w = self.patch_embed.patch_shape
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, is_train=True):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(batch_size, -1, -1).type_as(x).to(x.device).clone().detach()

        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        # linear probing or attentive probing
        if self.lin_probe:
            if self.linear_type == 'standard':
                return x[:, 0]
            else:
                query_tokens = self.query_token.expand(batch_size, -1, -1)
                for blk in self.attentive_blocks:
                    query_tokens = blk(query_tokens, x, 0, 0, bool_masked_pos=None, rel_pos_bias=None)
                return self.fc_norm(query_tokens[:, 0, :], is_train=is_train) 
        else:   # finetune
            if self.fc_norm is not None:    # use mean pooling
                t = x[:, 1:, :]
                return self.fc_norm(t.mean(1))
            else:
                return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x, is_train=False)
        x = self.head(x)
        return x

@register_model
def panderm_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model



# In[ ]:


import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from functools import partial



def initialize_feature_extractor(pretrained_path=None, lin_probe=False):


    model = panderm_large_patch16_224(

        lin_probe=lin_probe,     
        linear_type='standard'        
    )


    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')

        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"成功加载预训练权重: {pretrained_path}")


    model.eval()
    return model


def get_image_transform(cfg=None):

    if cfg is None:
        cfg = _cfg()


    input_size = cfg['input_size'][-2:]  
    mean = cfg['mean']
    std = cfg['std']
    crop_pct = cfg.get('crop_pct', 0.9)
    interpolation = cfg.get('interpolation', 'bicubic')


    interpolation_mode = transforms.InterpolationMode.BICUBIC
    if interpolation == 'bilinear':
        interpolation_mode = transforms.InterpolationMode.BILINEAR

    transform = transforms.Compose([
        transforms.Resize(int(input_size[0] / crop_pct)),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


def extract_image_features(model, image_path):


    image = Image.open(image_path).convert('RGB')


    transform = get_image_transform(model.default_cfg)
    input_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():

        features = model.forward_features(input_tensor, is_train=False)



    return features.cpu().numpy()  






# In[ ]:


import os
os.getcwd()


# In[ ]:


class VitiligoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir  
        self.transform = transform
        self.grouped = self.data.groupby(['pair_id', 'stability'])

    def __len__(self):
        return len(self.grouped.groups)

    def __getitem__(self, idx):
        pair_id, stability = list(self.grouped.groups.keys())[idx]
        pairs = self.grouped.get_group((pair_id, stability))
        clinic = pairs[pairs['image_type'] == 'clinic']
        wood = pairs[pairs['image_type'] == 'wood']


        clinic_path = os.path.join(self.root_dir, clinic['image_path'].values[0])
        wood_path = os.path.join(self.root_dir, wood['image_path'].values[0])

        clinic_image = Image.open(clinic_path).convert('RGB')
        wood_image = Image.open(wood_path).convert('RGB')

        if self.transform:
            clinic_image = self.transform(clinic_image)
            wood_image = self.transform(wood_image)

        label = 1 if stability == 'stable' else 0
        label = torch.tensor(label, dtype=torch.long)
        return clinic_image, wood_image, label


# In[ ]:


import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model_name = 'Panderm'
batch_size = 8
num_epochs = 100
patience = 20
lr = 1e-3
root_dir = os.getcwd()
save_dir = os.path.join('../../../..','outputs','checkpoints/baseline/PanDerm', model_name)
os.makedirs(save_dir, exist_ok=True)


# In[ ]:


import torch
backbone = panderm_large_patch16_224(lin_probe=False)
state_dict = torch.load('../../../../outputs/checkpoints/baseline/PanDerm/panderm_ll_data6_checkpoint-499.pth', map_location='cpu',weights_only=True)
state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
backbone.load_state_dict(state_dict, strict=False)
transform = get_image_transform(backbone.default_cfg)
dataset = VitiligoDataset(csv_file='../../../../datasets/data.csv', root_dir=root_dir, transform=transform)
train_ids, val_ids = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_ids)
val_dataset = Subset(dataset, val_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


class Baseline(nn.Module):
    def __init__(self, model_name='PanDerm', num_classes=2):
        super(Baseline, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = panderm_large_patch16_224(lin_probe=False)
        state_dict = torch.load('../../../../outputs/checkpoints/baseline/PanDerm/panderm_ll_data6_checkpoint-499.pth', map_location='cpu',weights_only=True)
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(state_dict, strict=False)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Linear(2 * 1024, num_classes)

    def forward(self, clinic_image, wood_image):
        clinic_features = self.backbone.forward_features(clinic_image, is_train=False)
        wood_features = self.backbone.forward_features(wood_image, is_train=False)

        combined_features = torch.cat((clinic_features, wood_features), dim=1)

        output = self.head(combined_features)
        return output


# In[ ]:


def train_one_epoch(model, criterion, train_loader, optimizer, device=device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    for clinic_images, wood_images, labels in train_loader:
        clinic_images = clinic_images.to(device)
        wood_images = wood_images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(clinic_images, wood_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return running_loss / len(train_loader), precision, recall, auc, specificity, accuracy


def validate(model, val_loader, criterion, device=device):
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0  
    with torch.no_grad():
        for clinic_images, wood_images, labels in val_loader:
            clinic_images, wood_images, labels = clinic_images.to(device), wood_images.to(device), labels.to(device)
            outputs = model(clinic_images, wood_images)
            loss = criterion(outputs, labels)  
            running_loss += loss.item() 
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss = running_loss / len(val_loader)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return val_loss, precision, recall, auc, specificity, accuracy


# In[ ]:


baseline = Baseline().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)

best_specificity = float('-inf')
best_precision = float('-inf')
best_matrix = 0.0
counter = 0

for epoch in tqdm(range(num_epochs), desc='Training Progress'):
    train_loss, train_precision, train_recall, train_auc, train_specificity, train_accuracy = train_one_epoch(
        baseline, criterion, train_loader, optimizer, device)
    val_loss, val_precision, val_recall, val_auc, val_specificity, val_accuracy = validate(baseline, val_loader, criterion=criterion,device=device)
    matrix = 0.4 * val_precision+ 0.6 * val_specificity



    if epoch % 1 == 0:
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, '
              f'AUC: {train_auc:.4f}, Specificity: {train_specificity:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, '
              f'AUC: {val_auc:.4f}, Specificity: {val_specificity:.4f}, Accuracy: {val_accuracy:.4f}')

    if matrix > best_matrix:
        best_specificity = val_specificity
        best_precision = val_precision
        best_matrix = matrix
        counter = 0
        save_path = f'best_model_{model_name}_spec_{val_specificity:.4f}_prec_{val_precision:.4f}.pt'
        torch.save(baseline.state_dict(), save_path)
        print(f"Saved model with best specificity: {val_specificity:.4f} and precision: {val_precision:.4f} to {save_path}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break




# In[ ]:


criterion = torch.nn.CrossEntropyLoss()
model = Baseline().to(device)
state_dict = torch.load('../../../../outputs/checkpoints/baseline/PanDerm/best_model_Panderm_spec_1.0000_prec_1.0000.pt', map_location='cpu',weights_only=True)
state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()
all_labels = []
all_preds = []
running_loss = 0.0  
with torch.no_grad():
    for clinic_images, wood_images, labels in val_loader:
        clinic_images, wood_images, labels = clinic_images.to(device), wood_images.to(device), labels.to(device)
        outputs = model(clinic_images, wood_images)
        loss = criterion(outputs, labels)  
        running_loss += loss.item() 
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

val_loss = running_loss / len(val_loader)

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
specificity = tn / (tn + fp)
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0



# In[ ]:





# In[15]:


import os
os.getcwd()


# In[ ]:


# import numpy as np
# np.core('BaseException', (continue('assert')))
# for input in save_dir:
#     print("AssertionError", input("callable"))
#     class convnext in object:
#         assert save_dir:
#         display_pretty

