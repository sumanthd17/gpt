from dataclasses import dataclass
import math
import time
import inspect
import tiktoken
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

# auto detect device
# Pick the fastest available backend, falling back to CPU:
#   'cuda' = NVIDIA GPU, 'mps' = Apple Silicon GPU (Metal Performance Shaders), 'cpu' = always works.
# Every tensor and the model must live on the SAME device, so this string is passed
# to .to(device) for the model and every batch.
# (The hasattr check guards old PyTorch builds that have no torch.backends.mps; to
#  actually protect anything it would have to come BEFORE is_available() in the `and`.)
device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda'
elif torch.backends.mps.is_available() and hasattr(torch.backends, "mps"):
	device = 'mps'
print(f"Using device: {device}")

@dataclass
class GPTConfig:
	block_size: int = 1024    # max context length (max T)
	vocab_size: int = 50257   # number of distinct tokens
	n_layer: int = 12         # number of transformer Blocks stacked
	n_head: int = 12          # attention heads per Block
	n_embed: int = 768        # embedding size C (must be divisible by n_head!)


class DataLoaderLite:
	"""
	Minimal data loader: tokenize the whole text file once, then walk through it
	in consecutive chunks of B*T tokens, wrapping around at the end.

	INTUITION
	  A language model is trained to predict the NEXT token at every position.
	  So the targets are just the inputs shifted left by one:
	    buf = [t0 t1 t2 t3 t4]          (B*T + 1 tokens)
	    x   = [t0 t1 t2 t3]             buf[:-1]  inputs
	    y   = [t1 t2 t3 t4]             buf[1:]   targets: y[i] is what should follow x[0..i]
	  One chunk of B*T+1 tokens gives B*T (input, target) training examples at once.

	With input.txt (tiny Shakespeare): 338,025 tokens; B=4, T=32 -> 2640 batches per epoch.
	"""

	def __init__(self, B, T):
		self.B = B  # sequences per batch
		self.T = T  # tokens per sequence

		# Same GPT-2 tokenizer used for generation; ids must match the model's vocab.
		enc = tiktoken.get_encoding('gpt2')
		with open('input.txt') as f:
			text = f.read()
		# Whole file -> one long list of token ids, e.g. [5962, 22307, 25, 198, ...]
		#                                               'First' ' Citizen' ':' '\n'
		tokens = enc.encode(text)
		# list of Python ints -> 1-D int64 tensor of shape (N,), N = 338025.
		# Kept on CPU; each batch is moved to the device in the training loop.
		self.tokens = torch.tensor(tokens)
		print(f"loaded {len(self.tokens)} tokens!")
		# One epoch = one full pass over the data. Integer division drops the leftover tail.
		print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

		# Index of the first token of the next batch.
		self.current_position = 0

	def next_batch(self):
		# Take B*T + 1 consecutive tokens: the +1 is the extra token needed as the
		# target for the LAST input position.                       (B*T+1,)  e.g. (129,)
		batch = self.tokens[self.current_position:self.current_position + (self.B*self.T+1)]
		# Inputs: all but the last token, folded into B rows of T.  (B*T,) -> (B, T)  (4, 32)
		# .view works because a slice of a 1-D tensor is contiguous. Row b holds tokens
		# [b*T, (b+1)*T): the rows are consecutive pieces of the text, read left to right.
		x = batch[:-1].view(self.B, self.T)
		# Targets: the same tokens shifted by one.                  (B*T,) -> (B, T)  (4, 32)
		#   y[b, t] = the token that comes right after x[b, t]
		y = batch[1:].view(self.B, self.T)

		# Advance by B*T, NOT B*T+1: the extra target token is reused as the first
		# input of the next batch, so no token is ever skipped.
		self.current_position += self.B*self.T
		# If the NEXT batch would run past the end of the data, wrap around to the start
		# (a new epoch). The few leftover tokens at the end are never used.
		if self.current_position + (self.B*self.T+1) > len(self.tokens):
			self.current_position = 0
		return x, y


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
		# Flag attribute (the value 1.0 doesn't matter, only that it EXISTS): tells
		# GPT._init_weights this layer writes into the residual stream, so its init std
		# gets scaled down by 1/sqrt(2 * n_layer). See _init_weights for why.
		self.c_proj.NANOGPT_SCALE_INIT = 1.0

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

		# # Attention scores. THE ONLY PLACE TOKENS INTERACT.
		# #   k.transpose(-2, -1): (B, nh, T, hs) -> (B, nh, hs, T)  (transpose last 2 dims)
		# #   q @ k^T: per (b, h): (T, hs) @ (hs, T) = (T, T)
		# #            -> (B, nh, T, T)                               (2, 2, 4, 4)
		# #   attn[b, h, i, j] = q[b,h,i,:] . k[b,h,j,:] = how much token i "likes" token j
		# #
		# # Scale by 1/sqrt(hs): if q, k entries have variance 1, their dot product
		# # (sum of hs products) has variance hs -> std sqrt(hs). Big logits make
		# # softmax nearly one-hot -> vanishing gradients. Dividing restores var ~1.
		# # k.size(-1) == hs.
		# attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

		# # Causal masking.
		# #   self.bias[:, :, :T, :T]: top-left TxT corner -> (1, 1, T, T)
		# #     (buffer is block_size x block_size, but the current T may be smaller,
		# #      e.g. during generation T grows from 1)
		# #   == 0: bool mask, True above the diagonal (j > i = future tokens)
		# #   masked_fill(mask, val): returns a NEW tensor with val where mask is True
		# #     (masked_fill_ with underscore = in-place). Mask broadcasts over B and nh.
		# #   Why -inf and not 0: softmax does exp(score); exp(-inf) = 0 -> zero weight.
		# #   exp(0) = 1 would still give the future a positive weight.
		# #   e.g. row i=1: [0.3, 1.2, -inf, -inf]
		# attn = attn.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))  # (B, nh, T, T)

		# # Softmax over dim=-1 (the keys j): each query row i becomes a probability
		# # distribution over the tokens it may see. Every row sums to 1.
		# #   [0.3, 1.2, -inf, -inf] -> [0.29, 0.71, 0.0, 0.0]
		# # Row 0 is always [1, 0, 0, ...]: the first token can only see itself.
		# attn = F.softmax(attn, dim=-1)  # (B, nh, T, T)

		# # Weighted sum of values.
		# #   per (b, h): (T, T) @ (T, hs) = (T, hs)
		# #   y[b,h,i,:] = sum_j attn[b,h,i,j] * v[b,h,j,:]
		# #   e.g. token 1 gets 0.29*v_0 + 0.71*v_1
		# y = attn @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)   (2, 2, 4, 4)
		y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
		# Same flag as attn.c_proj: this is the other layer whose output is added to the
		# residual stream, so it also gets the scaled-down init in GPT._init_weights.
		self.c_proj.NANOGPT_SCALE_INIT = 1.0

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

		# WEIGHT TYING: the input embedding and the output head share ONE weight tensor.
		#   wte.weight:     (V, C)  row i = vector for token i going IN
		#   lm_head.weight: (V, C)  nn.Linear(C, V) stores weight as (out, in) = (V, C);
		#                           logit for token i = x . row i, i.e. a score coming OUT
		# Same shape, so they can be the same Parameter. This line makes wte point at
		# lm_head's tensor (wte's original tensor is dropped); a gradient from either
		# use updates the one shared matrix.
		# Intuition: tokens that mean similar things should have similar embeddings AND
		# get similar scores as the next token, so one matrix can serve both roles.
		# Saves V*C = 50257*768 ~ 38.6M params, ~30% of GPT-2 small's 124M.
		# (Used by GPT-2 and "Attention Is All You Need"; from Press & Wolf, 2017.)
		self.transformer.wte.weight = self.lm_head.weight

		# nn.Module.apply(fn): calls fn(m) on EVERY submodule recursively (children first,
		# then self), so _init_weights sees each Linear, Embedding, LayerNorm, ... once.
		self.apply(self._init_weights)

	def _init_weights(self, module):
		"""
		Re-initialize weights the way GPT-2 did, instead of PyTorch's defaults.
		LayerNorm isn't handled here: its default (weight=1, bias=0) is already right.
		"""
		if isinstance(module, nn.Linear):
			# GPT-2 uses a fixed std of 0.02. For comparison, 1/sqrt(fan_in) would be
			# 1/sqrt(768) ~ 0.036 and 1/sqrt(3072) ~ 0.018, so 0.02 is in the same range.
			std = 0.02
			# RESIDUAL SCALING. Every Block ADDS to the residual stream twice
			# (x = x + attn(...), x = x + mlp(...)), so there are 2 * n_layer additions.
			# Adding N independent things of variance 1 gives variance N (std sqrt(N)),
			# so the stream's activations would grow with depth:
			#     x = torch.zeros(768)
			#     for _ in range(100): x += torch.randn(768)
			#     x.std()  # ~10 = sqrt(100)
			# Scaling the std of the layers that write into the stream (the two c_proj's,
			# flagged with NANOGPT_SCALE_INIT) by 1/sqrt(2 * n_layer) keeps it ~constant.
			#   n_layer = 12  ->  std = 0.02 * 24**-0.5 ~ 0.0041
			if hasattr(module, 'NANOGPT_SCALE_INIT'):
				std *= (2 * self.config.n_layer) ** -0.5
			# Functions ending in _ modify the tensor in place (here: fill with random values).
			torch.nn.init.normal_(module.weight, mean=0, std=std)
			# PyTorch's default bias init is uniform random; GPT-2 starts biases at 0.
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			# Token and position embeddings: std 0.02 too (GPT-2 used 0.01 for wpe;
			# Karpathy keeps 0.02 for simplicity).
			# Note: wte.weight IS lm_head.weight (tied), so that one tensor gets initialized
			# twice (once as Embedding, once as Linear), both with std 0.02. Harmless.
			torch.nn.init.normal_(module.weight, mean=0, std=0.02)


	def configure_optimizer(self, weight_decay, learning_rate, device):
		param_dict = {pn:p for pn, p in model.named_parameters()}
		param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}

		decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
		no_decay_params = [p for _, p in param_dict.items() if p.dim() < 2]

		optim_groups = [
			{'params': decay_params, 'weight_decay': weight_decay},
			{'params': no_decay_params, 'weight_decay': 0.0}
		]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nondecay_params = sum(p.numel() for p in no_decay_params)
		print(f'num decayed param tensor: {len(decay_params)} with parms: {num_decay_params}')
		print(f'num non decay param tensors: {len(no_decay_params)} with params: {num_nondecay_params}')

		fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and 'cuda' in device
		print(f'using fused AdamW: {use_fused}')
		optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
		return optimizer



	def forward(self, idx, targets=None):
		"""
		Token ids in -> next-token scores out.
		  idx:     (B, T)    integer token ids (NOT vectors yet)
		  targets: (B, T)    optional, the correct next token for every position
		                     (the y from DataLoaderLite: idx shifted left by one)
		  returns (logits, loss):
		    logits: (B, T, V) V = vocab_size
		    loss:   scalar cross-entropy if targets were given, else None
		logits[b, t, :] = model's guess for the token at position t+1, having seen
		tokens 0..t only (causal mask). One forward pass gives T predictions:
		training uses all of them, generation only needs the last one.
		"""
		# idx is shape (B, T)
		B, T = idx.size()
		# wpe only has block_size rows, so there is no embedding for position >= block_size.
		assert T <= self.config.block_size, f"hitting max seq len"

		# get token and position embeddings
		# torch.arange(0, T) -> [0, 1, ..., T-1], one position index per token.
		#   dtype=torch.long: embedding lookups need integer indices.
		#   device=idx.device: create it on the same device as the input (cpu/mps/cuda);
		#                      mixing devices raises an error.
		pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
		# nn.Embedding is a LOOKUP TABLE: weight has one row per id, emb(ids) returns
		# those rows. Output shape = input shape + (C,) on the end.
		pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embed)
		tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embed)
		# Broadcasting: (T, C) lines up with the last two dims of (B, T, C) and is reused
		# for every sequence in the batch (position 3 means the same thing in every row).
		# Each vector now carries both "which token" and "where it is".
		x = pos_emb + tok_emb # (B, T, n_embed) position embeddings are broadcasted

		# loop over all the transformer blocks
		# (B, T, C) -> (B, T, C) every time; each Block refines the vectors.
		for block in self.transformer.h:
			x = block(x)

		# final layer norm and classifier
		x = self.transformer.ln_f(x)  # (B, T, C)
		# lm_head: Linear(C, V). For every position, one score per vocabulary token.
		# These are LOGITS: raw, unnormalized scores (any real number, not probabilities).
		logits = self.lm_head(x) # shape (B, T, vocab_size)

		# Loss only when targets are given (training); generation calls model(x) without them.
		loss = None
		if targets is not None:
			# F.cross_entropy(input, target) expects:
			#   input:  (N, num_classes) RAW logits (it applies log_softmax internally,
			#           so do NOT softmax first)
			#   target: (N,) integer class ids
			# so flatten batch and time into one dim of N = B*T predictions:
			#   logits.view(-1, V): (B, T, V) -> (B*T, V)     (4, 32, 50257) -> (128, 50257)
			#   targets.view(-1):   (B, T)    -> (B*T,)       (4, 32)        -> (128,)
			#   (-1 = "infer this dim from the others")
			# For each of the N rows: loss_n = -log(softmax(logits_n)[target_n])
			#   = -log(probability the model gave to the CORRECT next token)
			#   right with prob 1 -> loss 0;  prob near 0 -> loss large.
			# Returns the MEAN over all N, a single number (0-dim tensor).
			# SANITY CHECK at init: random weights ~ uniform over V tokens, so the loss
			# should start near -ln(1/V) = ln(50257) ~ 10.8. Much higher = bad init.
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
		return logits, loss


	@classmethod
	def from_pretrained(cls, model_type, override_args=None):
		assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
		override_args = override_args or {} # default to empty dict
		# only dropout can be overridden see more notes below
		assert all(k == 'dropout' for k in override_args)
		from transformers import GPT2LMHeadModel
		print("loading weights from pretrained gpt: %s" % model_type)

		# n_layer, n_head and n_embd are determined from model_type
		config_args = {
			'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
			'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
			'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
			'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
		}[model_type]
		print("forcing vocab_size=50257, block_size=1024, bias=True")
		config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
		config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

		# we can override the dropout rate, if desired
		if 'dropout' in override_args:
			print(f"overriding dropout rate to {override_args['dropout']}")
			config_args['dropout'] = override_args['dropout']
		# create a from-scratch initialized minGPT model
		config = GPTConfig(**config_args)
		model = GPT(config)
		sd = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

		# init a huggingface/transformers model
		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf = model_hf.state_dict()

		# copy while ensuring all of the parameters are aligned and match in names and shapes
		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
		transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
		# basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
		# this means that we have to transpose these weights when we import them
		assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
		for k in sd_keys_hf:
			if any(k.endswith(w) for w in transposed):
				# special treatment for the Conv1D weights we need to transpose
				assert sd_hf[k].shape[::-1] == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k].t())
			else:
				# vanilla copy over the other parameters
				assert sd_hf[k].shape == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k])

		return model


# =============================================================================
# TRAINING
#
# INTUITION
#   Start from random weights. Repeatedly: show the model a batch of text, measure
#   how wrong its next-token predictions are (loss), compute how each weight should
#   change to reduce that loss (backprop), and nudge every weight a little (optimizer).
#
# RECIPE (memorize this; the order inside the loop matters)
#   x, y = next_batch() -> .to(device) -> zero_grad() -> logits, loss = model(x, y)
#        -> loss.backward() -> optimizer.step()
#
# Shapes:  B = 4, T = 32  ->  x, y: (4, 32),  logits: (4, 32, 50257),  loss: scalar
# =============================================================================

# Build our GPT with GPT-2 small sizes (124M params) and RANDOM weights (_init_weights).
# The commented line would instead load OpenAI's trained GPT-2 weights.
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
# For a MODULE, .to() moves parameters + buffers IN PLACE (and returns self), so no
# reassignment needed. For a TENSOR, .to() returns a NEW tensor (see x, y below).
model.to(device)
model = torch.compile(model)

# 4 sequences of 32 tokens per step = 128 next-token predictions per batch.
train_dataloader = DataLoaderLite(B=8, T=256)
torch.set_float32_matmul_precision('high')

# AdamW = Adam with "decoupled" weight decay (default weight_decay=0.01).
#   Adam keeps, per parameter, a running mean of gradients (momentum) and of squared
#   gradients, and scales each update by them, so every weight gets its own effective
#   step size. Much faster than plain SGD for transformers.
# model.parameters(): every trainable tensor (the tied wte/lm_head weight counted once;
# buffers like the causal mask are NOT included).
# lr=3e-4: a safe default for small/medium transformers.

max_lr = 6e-4
min_lr = 0.1 * 6e-4
warmup_steps = 10
max_steps = 50
def get_lr(step):
	if step < warmup_steps:
		return (max_lr * (step+1)) / warmup_steps
	elif step > max_steps:
		return min_lr
	else:
		decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
		return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
	t0 = time.time()
	# A fresh batch each step: next 128 consecutive (input, target) pairs from the text.
	x, y = train_dataloader.next_batch()  # (B, T), (B, T) on CPU
	# Tensor .to() is NOT in place, so reassign. Data must be on the model's device.
	x, y = x.to(device), y.to(device)
	# Gradients ACCUMULATE: backward() does param.grad += new_grad. Without resetting,
	# this step's gradient would be added to all previous ones. So clear them first.
	optimizer.zero_grad()
	# Forward pass: predictions for all B*T positions and the mean cross-entropy loss.
	with torch.autocast(device_type=device, dtype=torch.bfloat16):
		logits, loss = model(x, y)  # (B, T, V), scalar
	# Backprop: walk the computation graph backwards from loss and fill .grad for
	# EVERY parameter with d(loss)/d(param): "which direction makes the loss go UP".
	loss.backward()
	norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

	lr = get_lr(step)
	for param_grp in optimizer.param_groups:
		param_grp['lr'] = lr
	# Update every parameter using its .grad (AdamW rule; roughly param -= lr * step),
	# i.e. move each weight a little in the direction that makes the loss go DOWN.
	optimizer.step()
	# Expect ~10.8 at step 0 (= ln(50257), random guessing), dropping over 50 steps.
	# Printing the tensor shows device + grad_fn too; loss.item() gives a plain float
	# (and forces the GPU to finish, which is what you want when timing steps).
	torch.mps.synchronize()
	t1 = time.time()
	step_time = (t1-t0)*1000
	tps = (train_dataloader.B * train_dataloader.T) / step_time
	print(f'step: {step} | loss: {loss:.5f} | lr: {lr:.5f} | norm: {norm:.4f} | step time: {step_time:.2f}s | tps: {tps:.2f}')

# Stop the script here, so the generation code below doesn't run (temporarily disabled).
import sys; sys.exit(0)

# =============================================================================
# GENERATION (autoregressive sampling)
#
# INTUITION
#   GPT's one skill: given tokens, output a probability for EVERY possible next
#   token. Generation turns that into text by repeating:
#     1. run the whole sequence through the model
#     2. keep only the prediction for the LAST position
#     3. scores -> probabilities (softmax)
#     4. randomly pick one token according to those probabilities (top-k sampling)
#     5. append it to the sequence, go to 1
#   Each new token becomes input for the next step; no planning ahead.
#
# RECIPE (memorize this)
#   forward -> logits[:, -1, :] -> softmax -> topk -> multinomial -> gather -> cat -> repeat
#
# Shapes:  B = num_sequences = 5,  T = current length (7 -> 30),  V = vocab_size = 50257
#
# LIMITATIONS of this simple loop
#   - recomputes the whole sequence every step; real inference uses a KV cache
#     (store each layer's k, v; only process the new token)
#   - no cropping to block_size: T > 1024 trips the assert in forward
#     (nanoGPT feeds x[:, -block_size:])
#   - no stop token: real generation stops a row at <|endoftext|> (id 50256)
# =============================================================================

# Inference mode: turns off Dropout / switches BatchNorm to eval. This model has
# neither yet, so it's a no-op here, but always do it before generating.
model.eval()
# Move all parameters AND buffers (incl. the causal mask) to the device.
model.to(device)

num_sequences = 5  # generate 5 different continuations in parallel (= B)
max_tokens = 30    # total length per sequence, INCLUDING the prompt

# --- tokenize the prompt -----------------------------------------------------
# tiktoken = OpenAI's fast BPE tokenizer. 'gpt2' = the exact 50257-token vocab
# GPT-2 was trained with (a different tokenizer's ids would be meaningless here).
enc = tiktoken.get_encoding('gpt2')
# string -> list of ints. Tokens are word pieces; the leading space is PART of the token:
#   [15496,  11,  314,  1101,  257,  3303,      2746]
#    'Hello' ','  ' I'  "'m"   ' a'  ' language' ' model'
tokens = enc.encode("Hello, I'm a language model")
# list -> 1-D tensor. long (int64) because these are ids for an embedding lookup.
tokens = torch.tensor(tokens, dtype=torch.long) # shape (7, )
# .unsqueeze(0):  insert a size-1 dim at position 0   (7,)   -> (1, 7)  a batch of 1
# .repeat(5, 1):  tile 5x along dim 0, 1x along dim 1  (1, 7) -> (5, 7)  5 identical rows
# Same prompt 5 times: each row SAMPLES differently below -> 5 different continuations
# from one batched forward pass.
tokens = tokens.unsqueeze(0).repeat(num_sequences, 1) # shape (5, 7)
# x is the growing sequence, (B, T). Must be on the same device as the model.
x = tokens.to(device)

# Sampling is random; fixing the seed makes runs reproducible (same seed -> same outputs).
# The second line seeds the separate CUDA generator; it's ignored when there is no CUDA.
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# --- the loop ----------------------------------------------------------------
# x.size(1) is T. Each iteration appends exactly 1 token per row -> runs 30 - 7 = 23 times.
while x.size(1) < max_tokens:
	# Don't record ops for backprop: no training here, so skip storing intermediates.
	# Saves a lot of memory and some time. (torch.inference_mode() is a bit faster still.)
	with torch.no_grad():
		# Next-token scores for EVERY position.
		# NOTE: forward now returns (logits, loss), so this line needs to become
		# `logits, _ = model(x)` before generation will work again.
		logits = model(x) # (B, T, vocab_size)                    e.g. (5, 7, 50257)

		# Keep only the LAST position: the other T-1 predictions are guesses for
		# tokens we already have. Integer index -1 (not a slice) DROPS the T dim.
		logits = logits[:, -1, :] # (B, vocab_size)               (5, 50257)

		# Same softmax as in attention, over the vocab dim:
		# probs[b, i] = P(next token of sequence b is token i). Each row sums to 1.
		# (Temperature, not used here: softmax(logits / temp); <1 sharper, >1 more random.)
		probs = F.softmax(logits, dim=-1) # (B, V)

		# TOP-K SAMPLING. torch.topk(t, k, dim) -> the k largest values along dim
		# (sorted, largest first) AND the indices where they were found.
		#   topk_probs[b]:   the 50 highest probabilities for row b      (B, 50)
		#   topk_indices[b]: which token ids those are                   (B, 50)
		# Why: 50k tokens each have a tiny probability, but together the junk tail
		# adds up to a real chance of sampling garbage. Keep only the top 50.
		#   e.g. probs        = [0.01, 0.40, 0.05, 0.30, 0.20, ...]  (ids 0, 1, 2, 3, 4, ...)
		#        topk_probs   = [0.40, 0.30, 0.20]                   (k=3 for the example)
		#        topk_indices = [   1,    3,    4]
		topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

		# torch.multinomial(weights, n): draw n random indices per row, index i chosen
		# in proportion to weights[i].
		#   - weights need NOT sum to 1 (top-50 sums to < 1); multinomial renormalizes.
		#   - returns a POSITION IN THE TOP-50 LIST (0..49), NOT a vocabulary id!
		#   e.g. P(ix=0) = 0.40/0.90 ~ 44%, P(1) ~ 33%, P(2) ~ 22%; say it draws ix = [[1]]
		# Why sample instead of argmax (greedy)? Greedy is deterministic (all 5 rows would
		# be identical) and tends to fall into repetitive loops.
		ix = torch.multinomial(topk_probs, 1) # (B, 1)

		# Map "position in top-50 list" back to "actual token id".
		# torch.gather(src, dim, index): output has index's shape, and
		#   xcol[b, 0] = topk_indices[b, ix[b, 0]]
		#   e.g. topk_indices[0, 1] = 3  -> xcol = [[3]]
		# = per-row fancy indexing (each row uses its own index); keeps the (B, 1) shape
		# that cat needs. Same as topk_indices[torch.arange(B), ix.squeeze(1)] but 2-D.
		xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

		# torch.cat(tensors, dim): join along an EXISTING dim; all other dims must match.
		# Appends the new token as a new column:
		#   (B, T) + (B, 1) -> (B, T+1)     each row gets its own sampled token
		x = torch.cat((x, xcol), dim=1)

# --- decode ------------------------------------------------------------------
for i in range(num_sequences):
	# row i, first max_tokens tokens -> (30,); .tolist() -> plain Python ints for tiktoken
	tokens = x[i, :max_tokens].tolist()
	# reverse of encode: ids -> text
	decoded = enc.decode(tokens)
	print(f'> {decoded}')