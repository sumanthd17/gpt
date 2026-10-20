from dataclasses import dataclass
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda'
elif torch.backends.mps.is_available() and hasattr(torch.backends, "mps"):
	device = 'mps'
print(f"Using device: {device}")

@dataclass
class GPTConfig:
	block_size: int = 1024
	vocab_size: int = 50257
	n_layer: int = 12
	n_head: int = 12
	n_embed: int = 768


class DataLoaderLite:

	def __init__(self, B, T):
		self.B = B
		self.T = T

		enc = tiktoken.get_encoding('gpt2')
		with open('input.txt') as f:
			text = f.read()
		tokens = enc.encode(text)
		self.tokens = torch.tensor(tokens)
		print(f"loaded {len(self.tokens)} tokens!")
		print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

		self.current_position = 0

	def next_batch(self):
		batch = self.tokens[self.current_position:self.current_position + (self.B*self.T+1)]
		x = batch[:-1].view(self.B, self.T)
		y = batch[1:].view(self.B, self.T)

		self.current_position += self.B*self.T
		if self.current_position + (self.B*self.T+1) > len(self.tokens):
			self.current_position = 0
		return x, y


class CausalSelfAttention(nn.Module):

	def __init__(self, config):
		super().__init__()
		assert config.n_embed % config.n_head == 0

		self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

		self.c_proj = nn.Linear(config.n_embed, config.n_embed)
		self.c_proj.NANOGPT_SCALE_INIT = 1.0

		self.n_head = config.n_head
		self.n_embed = config.n_embed

		self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

	def forward(self, x):
		B, T, C = x.size()

		qkv = self.c_attn(x)

		q, k, v = qkv.split(self.n_embed, dim=2)

		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

		attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))

		attn = attn.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))

		attn = F.softmax(attn, dim=-1)

		y = attn @ v

		y = y.transpose(1, 2).contiguous().view(B, T, C)

		y = self.c_proj(y)

		return y


class MLP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
		self.gelu = nn.GELU(approximate='tanh')
		self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
		self.c_proj.NANOGPT_SCALE_INIT = 1.0

	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		return x


class Block(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embed)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embed)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class GPT(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.transformer = nn.ModuleDict(dict(
			wte = nn.Embedding(config.vocab_size, config.n_embed),
			wpe = nn.Embedding(config.block_size, config.n_embed),
			h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
			ln_f = nn.LayerNorm(config.n_embed)
		))
		self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

		self.transformer.wte.weight = self.lm_head.weight

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			std = 0.02
			if hasattr(module, 'NANOGPT_SCALE_INIT'):
				std *= (2 * self.config.n_layer) ** -0.5
			torch.nn.init.normal_(module.weight, mean=0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0, std=0.02)


	def forward(self, idx, targets=None):
		B, T = idx.size()
		assert T <= self.config.block_size, f"hitting max seq len"

		pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
		pos_emb = self.transformer.wpe(pos)
		tok_emb = self.transformer.wte(idx)
		x = pos_emb + tok_emb

		for block in self.transformer.h:
			x = block(x)

		x = self.transformer.ln_f(x)
		logits = self.lm_head(x)

		loss = None
		if targets is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
		return logits, loss


	@classmethod
	def from_pretrained(cls, model_type, override_args=None):
		assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
		override_args = override_args or {}
		assert all(k == 'dropout' for k in override_args)
		from transformers import GPT2LMHeadModel
		print("loading weights from pretrained gpt: %s" % model_type)

		config_args = {
			'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),
			'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024),
			'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280),
			'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600),
		}[model_type]
		print("forcing vocab_size=50257, block_size=1024, bias=True")
		config_args['vocab_size'] = 50257
		config_args['block_size'] = 1024

		if 'dropout' in override_args:
			print(f"overriding dropout rate to {override_args['dropout']}")
			config_args['dropout'] = override_args['dropout']
		config = GPTConfig(**config_args)
		model = GPT(config)
		sd = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf = model_hf.state_dict()

		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
		transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
		assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
		for k in sd_keys_hf:
			if any(k.endswith(w) for w in transposed):
				assert sd_hf[k].shape[::-1] == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k].t())
			else:
				assert sd_hf[k].shape == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k])

		return model


model = GPT(GPTConfig())
model.to(device)

train_dataloader = DataLoaderLite(B=4, T=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(50):
	x, y = train_dataloader.next_batch()
	x, y = x.to(device), y.to(device)
	optimizer.zero_grad()
	logits, loss = model(x, y)
	loss.backward()
	optimizer.step()
	print(f'step: {step} | loss: {loss}')

import sys; sys.exit(0)


model.eval()
model.to(device)

num_sequences = 5
max_tokens = 30

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_tokens:
	with torch.no_grad():
		logits = model(x)

		logits = logits[:, -1, :]

		probs = F.softmax(logits, dim=-1)

		topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

		ix = torch.multinomial(topk_probs, 1)

		xcol = torch.gather(topk_indices, -1, ix)

		x = torch.cat((x, xcol), dim=1)

for i in range(num_sequences):
	tokens = x[i, :max_tokens].tolist()
	decoded = enc.decode(tokens)
	print(f'> {decoded}')