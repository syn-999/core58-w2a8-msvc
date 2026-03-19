# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import os

import math
def apply_rotary_emb(xq, xk, theta=500000.0, offset_tensor=None):
    seq_len = xq.shape[1]
    head_dim = xq.shape[-1]
    freqs = (1.0 / (theta ** (torch.arange(0, head_dim, 2, device=xq.device).float() / head_dim)))
    t = torch.arange(seq_len, device=xq.device)
    if offset_tensor is not None:
        t = t + offset_tensor
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    def reshape_for_broadcast(freqs_cis, x):
        # freqs_cis is [seq_len, head_dim/2]
        # x is [bs, seq_len, n_local_kv_heads, heads_per_group, head_dim]
        shape = [1, freqs_cis.shape[0], 1, 1, freqs_cis.shape[1]]
        return freqs_cis.view(*shape)
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq)
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

import ctypes
from pathlib import Path
lib_ext = '.dll' if os.name == 'nt' else '.so'
current_dir = Path(__file__).parent
bitnet_lib_path = current_dir.parent / "src" / "cuda" / "bitnet_kernels" / f"libbitnet{lib_ext}"
bitnet_lib = ctypes.CDLL(str(bitnet_lib_path), winmode=0 if os.name == 'nt' else 0)

def bitnet_int8xint2_linear(input0, input1, s, ws):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=input0.device)

    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])

    return ret

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = False


LayerCache = Tuple[torch.Tensor, torch.Tensor]

class BitLinearKernel(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor
    weight_scale: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features//4, dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)

    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        return bitnet_int8xint2_linear(input, self.weight, s, self.weight_scale)

class BitLinear(nn.Linear):
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        input = self.quant_input(input)
        return F.linear(input, self.weight)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads

        Linear = BitLinearKernel if use_kernel else BitLinear

        self.wqkv = Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
        )
        self.wo = Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
        )

        self.attn_sub_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        xqkv = self.wqkv(x)
        # xqkv is [bs, seq_len, features]
        xq = xqkv[:, :, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, :, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, dim=-1)

        output_shape = xq.shape
        bs, seq_len = xq.shape[0], xq.shape[1]
        heads_per_group = self.n_local_heads // self.n_local_kv_heads
        xq = xq.view(
            bs, seq_len, self.n_local_kv_heads, heads_per_group, self.head_dim
        )
        # For xk, xv, we only have self.n_local_kv_heads in total, so we reshape to that directly
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, 1, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, 1, self.head_dim)

        # xq, xk are [1, seq_len, ...]
        seq_len = xq.shape[1]
        
        offset = attn_bias[0] - 1 if (attn_bias is not None and seq_len == 1) else None
        xq, xk = apply_rotary_emb(xq, xk, theta=self.rope_theta, offset_tensor=offset)
        
        # update cache
        cache_k, cache_v = cache
        
        if seq_len > 1:
            # Prefill: overwrite the start of the cache
            cache_k[:, :seq_len, ...] = xk
            cache_v[:, :seq_len, ...] = xv
            xk_used = xk
            xv_used = xv
            
            # Causal mask for prefill
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=xq.device)).unsqueeze(0).unsqueeze(0)
        else:
            # Decode: insert at offset
            pos = attn_bias[0] - 1
            
            # Use index_copy_ so we don't trigger CPU syncs with dynamic slice indices during CUDA recording
            pos_idx = pos.to(torch.int64).view(1)
            cache_k.index_copy_(1, pos_idx, xk)
            cache_v.index_copy_(1, pos_idx, xv)
            
            xk_used = cache_k
            xv_used = cache_v
            
            # Attention mask for valid tokens
            attn_mask = torch.arange(cache_k.shape[1], device=xq.device) <= pos
            attn_mask = attn_mask.view(1, 1, 1, -1)
        
        # GQA repeat across the n_kv_heads dimension to match heads_per_group
        xk_rep = xk_used.repeat_interleave(heads_per_group, dim=2)
        xv_rep = xv_used.repeat_interleave(heads_per_group, dim=2)
        
        # Collapse heads_per_group and n_local_kv_heads into n_local_heads
        # [bs, seq_len, n_local_heads, head_dim] -> transpose -> [bs, n_local_heads, seq_len, head_dim]
        xq = xq.view(bs, seq_len, -1, self.head_dim).transpose(1, 2)
        xk_rep = xk_rep.view(bs, xk_rep.shape[1], -1, self.head_dim).transpose(1, 2)
        xv_rep = xv_rep.view(bs, xv_rep.shape[1], -1, self.head_dim).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            xq, xk_rep, xv_rep,
            attn_mask=attn_mask,
        )
        
        output = output.transpose(1, 2).contiguous().view(output_shape)
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output

def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        Linear = BitLinearKernel if use_kernel else BitLinear

        self.w13 = Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)
        output = self.w2(inner)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        assert args.n_heads % n_kv_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attention.forward(
            self.attention_norm(x),
            cache,
            attn_bias,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

    @torch.no_grad()
    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        cache: list[LayerCache],
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], attn_bias)

        logits = self.output(self.norm(h))
        return logits.float()

    def forward(
        self,
        token_values: torch.Tensor,
        token_lengths: torch.Tensor,
        start_pos: torch.Tensor,
        cache: list[LayerCache],
        kv_padding: int,
    ) -> torch.Tensor:
        # Create standard boolean masks for causal functionality instead of xformers
        seq_len = token_lengths.item() if token_lengths.numel() == 1 else token_lengths[0].item()
        kv_seq_len = start_pos.item() + seq_len
        attn_bias = None
        if seq_len > 1:
            attn_bias = torch.tril(torch.ones(seq_len, kv_seq_len, dtype=torch.bool, device=token_values.device))
        
        return self.forward_with_attn_bias(token_values, attn_bias, cache)


def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[LayerCache]:

    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = args.n_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_kv_heads, 1, head_dim)
    
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: list[LayerCache], length: int) -> list[LayerCache]:
    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length

    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]