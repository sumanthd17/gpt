from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# Notation used in the shape comments throughout this file:
#   B  = batch size           (number of sequences processed in parallel)
#   T  = sequence length      (tokens per sequence, T <= block_size)
#   C  = n_embed              (embedding size / channels per token)
#   nh = n_head               (number of attention heads)
#   hs = C // nh              (head size, channels each head works with)
#
# Worked example used below:  B=2, T=4, C=8, nh=2, hs=4
# GPT-2 small for reference:  C=768, nh=12, hs=64, block_size=1024


@dataclass
class GPTConfig:
    block_size: int = 256  # max context length (max T)
    vocab_size: int = 65   # number of distinct tokens
    n_layer: int = 6       # number of transformer Blocks stacked
    n_head: int = 8        # attention heads per Block
    n_embed: int = 384     # embedding size C (must be divisible by n_head!)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    INTUITION
      Every token emits three vectors:
        query (q): "what am I looking for?"
        key   (k): "what do I contain?"
        value (v): "what will I hand over if you attend to me?"
      For token i:
        score[i, j]  = q_i . k_j                 (how well j matches what i wants)
        weight[i, :] = softmax(score[i, :])      (only over j <= i -> causal)
        y_i          = sum_j weight[i, j] * v_j  (weighted average of values)
      Multi-head = do this nh times in parallel, each head on its own hs-sized
      slice of the channels, so different heads can look for different things.

    RECIPE (memorize this)
      1. project   x -> q, k, v           (one Linear C -> 3C, then split)
      2. split heads                      view(B,T,nh,hs).transpose(1,2)
      3. scores    q @ k^T / sqrt(hs)     -> (B, nh, T, T)
      4. mask      future -> -inf
      5. softmax   over last dim (keys)
      6. mix       attn @ v               -> (B, nh, T, hs)
      7. merge heads                      transpose(1,2).contiguous().view(B,T,C)
      8. project   c_proj
      (steps 3-6 == F.scaled_dot_product_attention(q, k, v, is_causal=True))
    """

    def __init__(self, config):
        super().__init__()
        # C channels get cut into nh equal pieces of size hs, so it must divide evenly.
        assert config.n_embed % config.n_head == 0

        # q, k, v projections fused into ONE Linear layer.
        #   weight: (3C, C), bias: (3C,)   rows [0:C]->q, [C:2C]->k, [2C:3C]->v
        #   Same math as three separate nn.Linear(C, C), but one big matmul is
        #   faster on GPU, and it matches the name/layout in OpenAI's GPT-2 weights.
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

        # Output projection (C -> C). After attention each head's output lives in
        # its own block of channels; concatenating doesn't mix them. c_proj is
        # where information from the different heads gets combined.
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Causal mask. Built inside-out:
        #   torch.ones(bs, bs)   -> (bs, bs) all ones
        #   torch.tril(...)      -> keep lower triangle (incl. diagonal), zero the rest
        #                             j=0 j=1 j=2 j=3
        #                       i=0 [  1   0   0   0 ]   row i = query token
        #                       i=1 [  1   1   0   0 ]   col j = key token
        #                       i=2 [  1   1   1   0 ]   1 = "i may look at j"
        #                       i=3 [  1   1   1   1 ]
        #   .view(1, 1, bs, bs)  -> add two size-1 dims so it broadcasts
        #                           against attention scores of shape (B, nh, T, T)
        # register_buffer: part of module state (moves with .to(device), saved in
        # state_dict) but NOT a parameter (no grad, optimizer never touches it).
        # Called "bias" only to match OpenAI/HF checkpoint naming; it's really a mask.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # x: (B, T, C)                                         e.g. (2, 4, 8)
        B, T, C = x.size()  # .size() == .shape

        # Linear acts on the LAST dim only, so every token is projected
        # independently with the same weights. No token has seen another yet.
        qkv = self.c_attn(x)  # (B, T, C) -> (B, T, 3C)         (2, 4, 24)

        # split(split_SIZE, dim): chunks of size C along dim 2 -> 3 chunks.
        # (contrast: torch.chunk(qkv, 3, dim=2) takes the NUMBER of chunks)
        q, k, v = qkv.split(self.n_embed, dim=2)  # 3 x (B, T, C)  3 x (2, 4, 8)

        # Split into heads, then move the head dim in front of T.
        #
        # .view(B, T, nh, hs): (B, T, C) -> (B, T, nh, hs)        (2, 4, 2, 4)
        #   Reinterprets the same memory (no copy). The C channels of a token
        #   are cut into nh contiguous blocks of hs:
        #     [c0 c1 c2 c3 | c4 c5 c6 c7]
        #      └─ head 0 ─┘ └─ head 1 ─┘
        #   i.e. new[b, t, h, d] = old[b, t, h*hs + d]
        #
        # .transpose(1, 2): (B, T, nh, hs) -> (B, nh, T, hs)       (2, 2, 4, 4)
        #   Why: `@` (torch.matmul) multiplies the LAST TWO dims as matrices and
        #   treats ALL leading dims as batch dims. We want, for every (b, h),
        #   a (T, hs) matrix -> heads must sit before T so they act like batch.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Attention scores. THE ONLY PLACE TOKENS INTERACT.
        #   k.transpose(-2, -1): (B, nh, T, hs) -> (B, nh, hs, T)  (transpose last 2 dims)
        #   q @ k^T: per (b, h): (T, hs) @ (hs, T) = (T, T)
        #            -> (B, nh, T, T)                               (2, 2, 4, 4)
        #   attn[b, h, i, j] = q[b,h,i,:] . k[b,h,j,:] = how much token i "likes" token j
        #
        # Scale by 1/sqrt(hs): if q, k entries have variance 1, their dot product
        # (sum of hs products) has variance hs -> std sqrt(hs). Big logits make
        # softmax nearly one-hot -> vanishing gradients. Dividing restores var ~1.
        # k.size(-1) == hs.
        attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        # Causal masking.
        #   self.bias[:, :, :T, :T]: top-left TxT corner -> (1, 1, T, T)
        #     (buffer is block_size x block_size, but the current T may be smaller,
        #      e.g. during generation T grows from 1)
        #   == 0: bool mask, True above the diagonal (j > i = future tokens)
        #   masked_fill(mask, val): returns a NEW tensor with val where mask is True
        #     (masked_fill_ with underscore = in-place). Mask broadcasts over B and nh.
        #   Why -inf and not 0: softmax does exp(score); exp(-inf) = 0 -> zero weight.
        #   exp(0) = 1 would still give the future a positive weight.
        #   e.g. row i=1: [0.3, 1.2, -inf, -inf]
        attn = attn.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))  # (B, nh, T, T)

        # Softmax over dim=-1 (the keys j): each query row i becomes a probability
        # distribution over the tokens it may see. Every row sums to 1.
        #   [0.3, 1.2, -inf, -inf] -> [0.29, 0.71, 0.0, 0.0]
        # Row 0 is always [1, 0, 0, ...]: the first token can only see itself.
        attn = F.softmax(attn, dim=-1)  # (B, nh, T, T)

        # Weighted sum of values.
        #   per (b, h): (T, T) @ (T, hs) = (T, hs)
        #   y[b,h,i,:] = sum_j attn[b,h,i,j] * v[b,h,j,:]
        #   e.g. token 1 gets 0.29*v_0 + 0.71*v_1
        y = attn @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)   (2, 2, 4, 4)

        # Merge heads back (reverse of the split above).
        #   .transpose(1, 2):  (B, nh, T, hs) -> (B, T, nh, hs)         (2, 4, 2, 4)
        #   .contiguous():     transpose only changes STRIDES, not memory order.
        #                      .view() needs memory laid out in the new shape's
        #                      order, so copy into a fresh, ordered block first.
        #                      (.reshape(B, T, C) = contiguous-if-needed + view)
        #   .view(B, T, C):    (B, T, nh, hs) -> (B, T, C)              (2, 4, 8)
        #                      == concatenating head outputs side by side:
        #                      [ head0 out (hs) | head1 out (hs) ]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Mix information across heads. Output shape == input shape, which is what
        # allows the residual connection x = x + attn(...) in Block.
        y = self.c_proj(y)  # (B, T, C)

        return y



class MLP(nn.Module):
    """
    Position-wise feed-forward network. Attention lets tokens COMMUNICATE;
    the MLP lets each token THINK about what it gathered, independently
    (no interaction between tokens here, Linear acts on the last dim only).
    """

    def __init__(self, config):
        super().__init__()
        # Expand to 4x width (C -> 4C), as in the original Transformer / GPT-2.
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        # GELU non-linearity; 'tanh' approximation is what GPT-2 used.
        # nn.GELU is a MODULE (call later with x); F.gelu is a FUNCTION (needs x now).
        self.gelu = nn.GELU(approximate='tanh')
        # Project back down (4C -> C) so the output can be added to the residual.
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        # x: (B, T, C)
        x = self.c_fc(x)    # (B, T, 4C)
        x = self.gelu(x)    # (B, T, 4C)
        x = self.c_proj(x)  # (B, T, C)
        return x


class Block(nn.Module):
    """
    One transformer block: communicate (attention), then compute (MLP).

    Pre-norm + residual (GPT-2 style):
        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))
    - LayerNorm is applied to the INPUT of each sub-layer, not to the residual.
    - The residual path x flows through untouched ("gradient superhighway"):
      addition passes gradients straight through, which makes deep stacks trainable.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        # x: (B, T, C) throughout; every sub-layer preserves the shape.
        x = x + self.attn(self.ln_1(x))  # tokens exchange information
        x = x + self.mlp(self.ln_2(x))   # each token processes what it gathered
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Module names mirror HF's GPT-2 so pretrained weights can be loaded by name.
        self.transformer = nn.ModuleDict(dict(
            # token embedding:    token id (0..vocab_size-1) -> C-dim vector. weight (vocab_size, C)
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            # position embedding: position (0..block_size-1) -> C-dim vector. weight (block_size, C)
            # Attention itself is order-agnostic; this is how the model knows token order.
            wpe = nn.Embedding(config.block_size, config.n_embed),
            # the stack of n_layer transformer Blocks
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            # final LayerNorm (needed because pre-norm leaves the last residual un-normalized)
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        # Language-model head: C -> vocab_size logits (a score for every possible next token).
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)


cfg = GPTConfig(block_size=8, n_embed=8, n_head=2)
attn = CausalSelfAttention(cfg)
x = torch.randn(2, 4, 8)

# 1) Print every intermediate shape (add prints inside forward) and check them against the table.
print(attn(x).shape)  # torch.Size([2, 4, 8])

# 2) Causality test: changing a future token must not change earlier outputs.
x2 = x.clone()
x2[:, 3] = torch.randn(2, 8)          # change only the last token
y1, y2 = attn(x), attn(x2)
print(torch.allclose(y1[:, :3], y2[:, :3]))  # True: tokens 0-2 can't see token 3
print(torch.allclose(y1[:, 3], y2[:, 3]))    # False: token 3 changed
