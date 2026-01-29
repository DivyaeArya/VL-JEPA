
# Copyright 2025 J Joe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
import time
import math
from types import SimpleNamespace
import requests
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from safetensors.torch import load_file
from PIL import Image

CFG_L = dict(rms_norm_eps = 1e-6, rope_base = 10000.0, attn_bias = False)
CFG_V = dict(image_size = 224, num_channels = 3, layer_norm_eps = 1e-6, attn_bias = True)
URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, alpha, scale, dropout):
        output_dims, input_dims = linear.weight.shape
        # Note: PyTorch Linear weight is (out_features, in_features)
        lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, scale=scale, dropout=dropout, bias=(linear.bias is not None))
        lora_lin.linear = linear # Share the linear layer (or replace it, here we wrap it)
        # Re-initialize m to match the wrapped linear weight
        with torch.no_grad():
             lora_lin.m.copy_(torch.linalg.norm(linear.weight, dim=1))
        return lora_lin

    def __init__(self, input_dims, output_dims, r, alpha, scale, dropout, bias=False):
        super().__init__()
        # self.linear will be assigned later or initialized here if used standalone
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        
        # LoRA weights
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = nn.Parameter(torch.empty(r, input_dims))
        nn.init.uniform_(self.lora_a, -init_scale, init_scale)
        self.lora_b = nn.Parameter(torch.zeros(output_dims, r))
        
        # Magnitude vector m
        self.m = nn.Parameter(torch.zeros(output_dims))
        # Initialized in from_linear or manually

    def _get_weight_norm(self):
        # Calculate the norm of the adapted weight (W0 + BA)
        w = self.linear.weight
        # W_adapted = W + scale * B @ A. 
        # In PyTorch Linear: y = x @ W.T + b.  Weight shape is (out, in).
        # So W_adapted = W + scale * (B @ A) 
        # B shape: (out, r), A shape: (r, in). B @ A shape: (out, in).
        return w + (self.scale * self.lora_b @ self.lora_a)

    def forward(self, x):
        # DoRA logic:
        # 1. Calculate adapted direction: V = W0 + BA
        # 2. Normalize V: V / ||V||
        # 3. Apply magnitude m: W' = m * (V / ||V||)
        # 4. y = x @ W'.T + bias
        
        # Optimised forward (similar to MLX implementation):
        # y = linear(x)
        # z = (dropout(x) @ lora_a.T) @ lora_b.T
        # z = y + scale * z  <-- This is effectively x @ (W0 + BA).T
        # So z is the pre-activation output using the un-normalized adapted weight.
        
        # Then we need to correct the magnitude.
        # W_adapted = W0 + scale * B @ A
        # current_norm = ||W_adapted|| (row-wise)
        # target_norm = m
        # scale_factor = m / current_norm
        # final_out = scale_factor * z
        
        y = self.linear(x) # x @ W.T + b
        
        # lora path: x @ A.T @ B.T
        # lora_a: (r, in), lora_b: (out, r)
        dropout_x = self.dropout(x)
        lora_out = dropout_x @ self.lora_a.t() @ self.lora_b.t()
        
        z = y + (self.scale * lora_out) # Contribution from W + BA (plus bias)
        
        # To strictly follow DoRA, we normalize z by the norm of the adapted weight, then multiply by m.
        # Note: z includes bias. The magnitude scaling should happen on the weight contribution only?
        # MLX code: 
        # y = y - bias
        # ... DoRA stuff on z (which is linear output w/o bias + lora output)
        # z = (m / denom) * z
        # z = z + bias
        
        bias = self.linear.bias if self.linear.bias is not None else 0
        if isinstance(bias,  torch.Tensor):
             z_no_bias = z - bias
        else:
             z_no_bias = z
             
        w_adapted = self._get_weight_norm()
        denom = torch.linalg.norm(w_adapted, dim=1) + 1e-6
        
        # Broadcast m / denom
        factor = (self.m / denom).view(-1) # shape (out_features)
        
        z_out = z_no_bias * factor # Broadcasting over last dim
        
        if isinstance(bias, torch.Tensor):
            z_out = z_out + bias
            
        return z_out

def to_dora(layers, targets=None, rank=8, scale=0.1):
    _targets = ['o_proj', 'down_proj'] if targets is None else targets
    for layer in layers:
        for name, module in layer.named_modules():
            if any(name.endswith(t) for t in _targets):
                 if isinstance(module, nn.Linear):
                    # We need to replace the module in the layer
                    # Finding parent module
                    path = name.split('.')
                    parent = layer
                    for p in path[:-1]:
                        parent = getattr(parent, p)
                    target_name = path[-1]
                    
                    original_linear = getattr(parent, target_name)
                    dora_layer = DoRALinear.from_linear(original_linear, r=rank, alpha=rank, scale=scale, dropout=0.0)
                    setattr(parent, target_name, dora_layer)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = config.hidden_size
        bias = config.attn_bias
        self.n_heads = n_heads = config.num_attention_heads
        head_dim = dims // n_heads
        self.n_kv_heads = n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        self.scale = head_dim**-0.5
        
        self.q_proj = nn.Linear(dims, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dims, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dims, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dims, bias=bias)
        
        self.rope_base = getattr(config, 'rope_base', False)
        
    def _apply_rope(self, x, offset=0):
        # x: [B, H, L, D]
        # Simple RoPE implementation
        if not self.rope_base:
            return x
        
        B, H, L, D = x.shape
        # Create position indices
        positions = torch.arange(offset, offset + L, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
        freqs = torch.outer(positions, inv_freq) # [L, D/2]
        
        emb = torch.cat((freqs, freqs), dim=-1) # [L, D]
        
        # rotate
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        
        return x * emb.cos() + rotated_x * emb.sin()

    def forward(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        head_dim = (self.q_proj.weight.shape[0] // self.n_heads)
        
        queries = self.q_proj(x).view(B, L, self.n_heads, -1).transpose(1, 2) # [B, n_heads, L, head_dim]
        keys = self.k_proj(x).view(B, L, self.n_kv_heads, -1).transpose(1, 2)
        values = self.v_proj(x).view(B, L, self.n_kv_heads, -1).transpose(1, 2)
        
        if cache is not None:
             key_cache, value_cache = cache
             queries = self._apply_rope(queries, offset=key_cache.shape[2])
             keys = self._apply_rope(keys, offset=key_cache.shape[2])
             keys = torch.cat([key_cache, keys], dim=2)
             values = torch.cat([value_cache, values], dim=2)
        else:
             queries = self._apply_rope(queries)
             keys = self._apply_rope(keys)
        
        # Scaled Dot Product Attention
        # PyTorch SDPA expects [B, H, L, D] (or similar broadcastable)
        output = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask, scale=self.scale)
        
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(output), (keys, values)

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.projection_dim, bias=True)

    def forward(self, x):
        return self.linear(x)

class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Input channel logic handled by Conv2d
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.num_patches = (config.image_size // config.patch_size) ** 2 
        self.position_embedding = nn.Embedding(self.num_patches, config.hidden_size)
    
    def forward(self, x):
        # x is [B, C, H, W] for PyTorch
        patches = self.patch_embedding(x) # [B, D, H', W']
        patches = patches.flatten(2) # [B, D, L]
        patches = patches.transpose(1, 2) # [B, L, D]
        
        pos_ids = torch.arange(self.num_patches, device=x.device).unsqueeze(0)
        return patches + self.position_embedding(pos_ids)

class GELU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GELU(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        r, _ = self.self_attn(self.layer_norm1(x))
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r

class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = self.embeddings(x)
        for l in self.layers:
            x = l(x)
        x = self.post_layernorm(x) 
        return x

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def forward(self, x):
        # Using PyTorch 2.0+ RMSNorm if available, otherwise manual.
        # Manual implementation:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)

class GeGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = GeGLU(config)
        self.input_layernorm = RMSNorm(config)
        self.post_attention_layernorm = RMSNorm(config)

    def forward(self, x, mask = None, cache = None):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache

class LanguageModel(nn.Module):
    def __init__(self, config, num_layers_override=None):
        super().__init__()
        self.scale = config.hidden_size**0.5
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        n_layers = config.num_hidden_layers if num_layers_override is None else num_layers_override
        self.layers = nn.ModuleList([TransformerBlock(config=config) for _ in range(n_layers)])
        self.norm = RMSNorm(config)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask_4d=None, cache=None, output_hidden_states=False):
        cache = [None] * len(self.layers) if cache is None else cache
        if inputs_embeds is None:
             h = self.embed_tokens(input_ids)
        else:
             h = inputs_embeds
             
        h = h * self.scale
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, attention_mask_4d, cache[e])
        
        if output_hidden_states:
            return self.norm(h)
        # return logits (linear projection from embedding weights)
        # In PyTorch, we can do F.linear with the embedding weight
        return F.linear(self.norm(h), self.embed_tokens.weight), cache

class PG(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        model_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "*.json"], token=os.getenv('HF_TOKEN'))
        config = _get_cfg(f"{model_path}/config.json")
        config.vision_config = SimpleNamespace(**(CFG_V|config.vision_config))
        config.text_config = SimpleNamespace(**(CFG_L|config.text_config))
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Projector(config.vision_config)
        _get_wt(model_path, config, model=self)

class X_Encoder(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = Projector(config) 
    
    def forward(self, x_v):
        x_v = self.vision_tower(x_v) 
        x_v = self.multi_modal_projector(x_v)
        return x_v

class Y_Encoder(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.language_model = LanguageModel(config, num_layers_override=num_layers)
    def forward(self, y):
        # y: input_ids
        y = self.language_model(input_ids=y, output_hidden_states=True)
        return torch.mean(y, dim=1)

class Predictor(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.language_model = LanguageModel(config, num_layers_override=num_layers)
    def forward(self, s_v, query_ids):
        s_v = s_v / self.language_model.scale
        x_q = self.language_model.embed_tokens(query_ids)
        x = torch.cat([s_v, x_q], dim=1)
        x = self.language_model(input_ids=None, inputs_embeds=x, output_hidden_states=True)
        return torch.mean(x, dim=1)

class Y_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = LanguageModel(config)
    
    @torch.no_grad()
    def forward(self, s_y):
        logits, cache = self.language_model(input_ids=None, inputs_embeds=s_y)
        token = torch.argmax(logits[:, -1, :], dim=-1)
        list_tokens = token.tolist()
        for _ in range(100):
            logits, cache = self.language_model(input_ids=token.unsqueeze(0), cache=cache)
            token = torch.argmax(logits[:, -1, :], dim=-1)
            list_tokens += token.tolist()
            # if list_tokens[-1] == processor.tokenizer.eos_token_id:
            #     break
        return list_tokens

class VLJEPA(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        model_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "*.json"], token=os.getenv('HF_TOKEN'))
        config = _get_cfg(f"{model_path}/config.json")
        config.vision_config = SimpleNamespace(**(CFG_V|config.vision_config))
        config.text_config = SimpleNamespace(**(CFG_L|config.text_config))
        self.config=config
        self.x_encoder = X_Encoder(config.vision_config, num_layers=4)
        self.y_encoder = Y_Encoder(config.text_config, num_layers=4)
        self.predictor = Predictor(config.text_config, num_layers=4)
        self.y_decoder = Y_Decoder(config.text_config)
        _get_wt(model_path, config, model=self.x_encoder)
        _get_wt(model_path, config, model=self.y_encoder)
        _get_wt(model_path, config, model=self.predictor)
        _get_wt(model_path, config, model=self.y_decoder)
    
    def forward(self, pixel_values, query_ids):
        s_v = self.x_encoder(pixel_values)
        return self.predictor(s_v, query_ids)

def _get_cfg(json_path, **kwargs):
    try:
        with open(json_path, "r") as f:
            cfg = SimpleNamespace(**(json.load(f)|kwargs))
        return cfg
    except:
        return False

def _get_wt(model_path, model_cfg, model=None):
    # Determine loading strategy
    # The weights from HF (safetensors) are typically NCHW for vision
    wt = {}
    for wf in glob.glob(f"{model_path}/*.safetensors"):
        state_dict = load_file(wf)
        for k, v in state_dict.items():
            new_k = k.replace('vision_tower.vision_model.', 'vision_tower.') \
                     .replace('language_model.model.', 'language_model.') \
                     .replace('encoder.layers.', 'layers.') \
                     .replace('self_attn.out_proj.','self_attn.o_proj.')
            # No transpose needed for PyTorch Conv2d if source is standard HF
            wt[new_k] = v

    if model is None:
        return wt
    
    # Filter and load
    model_keys = set(model.state_dict().keys())
    # Partial load logic similar to MLX's fuzzy matching
    
    # We create a filtered dict
    filtered_wt = {}
    for k, v in wt.items():
        # Check if this key belongs to the model (roughly)
        # MLX logic was: any(i[0].startswith(_k) for _k in ok) where ok is model keys.
        # This allows loading "layers.0..." into "layers.0..."
        # But we must be careful about parameter names.
        
        # In PyTorch, strict matching is usually better, but here we might have mismatched prefixes
        # The replacement rules above handle most prefix fixes.
        
        # We try to match with model keys.
        # Iterate model keys and see if 'k' matches relevant part.
        pass

    # Actually, simpler approach:
    # Iterate over model named_parameters, find matching key in wt
    
    final_state_dict = {}
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    all_model_keys = set(model_params.keys()) | set(model_buffers.keys())
    
    for mk in all_model_keys:
        # mk might be "vision_tower.embeddings.patch_embedding.weight"
        # wt might have "vision_tower.embeddings.patch_embedding.weight"
        
        # We need to handle the case where we load a subset of layers (e.g. layers 0-3 for encoder).
        # The MLX code creates fresh encoders with `num_layers=4`.
        # The weights `layers.0`..`layers.31` exist in file.
        # We need to map `layers.0` to `layers.0`.
        
        if mk in wt:
            final_state_dict[mk] = wt[mk]
        else:
            # Fallback for layer index mismatch?
            # MLX code `_get_wt` filters keys that STARTWITH model keys. 
            # If model has `layers.0`, and wt has `layers.0`, it matches.
            pass
            
    # Load what we found
    model.load_state_dict(final_state_dict, strict=False)

def infonce_loss(predicted_embeddings, target_embeddings, temperature=0.07):
    # predicted_embeddings: [B, D]
    # target_embeddings: [B, D]
    
    pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
    target_norm = F.normalize(target_embeddings, p=2, dim=-1)
    
    logits = (pred_norm @ target_norm.T) / temperature
    B = logits.shape[0]
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, labels)

def train_vljepa(model, dataset_iterator, n_epoch=2, lr=1e-4):
    # Freeze model
    for p in model.parameters():
        p.requires_grad = False
    
    # Apply DoRA
    to_dora(model.predictor.language_model.layers)
    
    # Function to enable grads for DoRA parts
    # to_dora adds parameters. 
    # DoRALinear registers lora_a, lora_b, m as parameters.
    # We should ensure they require grad. They do by default when created.
    # But we called freeze() on the whole model before!
    # Wait, `to_dora` is called AFTER freeze. So new params are created fresh (requires_grad=True default).
    # BUT `model.predictor...` are existing modules. `to_dora` modifies them in-place.
    # The new DoRALinear creates new Parameters.
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    device = next(model.parameters()).device
    model.train()
    
    for epoch in range(n_epoch):
        total_loss = 0
        steps = 0
        tic = time.perf_counter()
        
        for batch in dataset_iterator():
            images, queries, answers = batch
            # Convert to tensor/device if needed
            images = torch.from_numpy(images).to(device) if isinstance(images, np.ndarray) else images.to(device)
            queries = torch.from_numpy(queries).to(device) if isinstance(queries, np.ndarray) else queries.to(device)
            answers = torch.from_numpy(answers).to(device) if isinstance(answers, np.ndarray) else answers.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            target_emb = model.y_encoder(answers)
            pred_emb = model(images, queries)
            
            loss = infonce_loss(pred_emb, target_emb)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            if steps >= 10: break
            
        print(f"Epoch {epoch+1}: Loss {total_loss/steps:.4f} ({time.perf_counter() - tic:.2f}s)")

def get_overfit_batch(processor, batch_size=2):
    try:
        car_image = Image.open(requests.get(URL, stream=True).raw)
    except:
        # Fallback if offline or URL fails
        car_image = Image.new('RGB', (224, 224), color='red')

    car_text = "A car"
    noise_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    noise_text = "Noise"
    query_text = "Caption: What is this?"
    
    while True:
        imgs = [car_image, noise_image]
        pixel_values = processor.image_processor(imgs, return_tensors="np").pixel_values
        # pixel_values shape: (B, C, H, W). PyTorch convention.
        
        q_tokens = processor.tokenizer([query_text, query_text], return_tensors="np", padding=True).input_ids
        a_tokens = processor.tokenizer([car_text, noise_text], return_tensors="np", padding=True).input_ids
        
        yield pixel_values, q_tokens, a_tokens

def test_sanity(model_id="google/paligemma-3b-mix-224"):
    print("Running Sanity Check...")
    model = PG(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        raw_image = Image.open(requests.get(URL, stream=True).raw)
    except:
        raw_image = Image.new('RGB', (224, 224), color='red')

    print(f'{raw_image=}')
    pixel_values = processor.image_processor(raw_image, return_tensors="pt").pixel_values.to(device)
    # [1, 3, 224, 224]
    
    with torch.no_grad():
        vis_features = model.vision_tower(pixel_values)
        vis_features = model.multi_modal_projector(vis_features)
        vis_features = vis_features / model.language_model.scale
        
        prompt = "What is this?"
        input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        text_embeds = model.language_model.embed_tokens(input_ids) # [1, L, D]
        curr_input = torch.cat([vis_features, text_embeds], dim=1)
        
        print(f"Prompt: {prompt}")
        print("Output: ", end="", flush=True)
        
        # Generate
        # Initial pass
        logits, cache = model.language_model(
            input_ids=None,
            inputs_embeds=curr_input,
            cache=None
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        print(processor.decode(next_token), end="", flush=True)
        
        for i in range(20):
             curr_input = model.language_model.embed_tokens(next_token.unsqueeze(1))
             logits, cache = model.language_model(
                 input_ids=None,
                 inputs_embeds=curr_input,
                 cache=cache
             )
             next_token = torch.argmax(logits[:, -1, :], dim=-1)
             token_id = next_token.item()
             if token_id == processor.tokenizer.eos_token_id:
                 break
             print(processor.decode([token_id]), end="", flush=True)
    print("\nSanity Check Done.")

def test_retrieval(model, processor):
    print("Running Retrieval Test...")
    device = next(model.parameters()).device
    try:
        raw_image = Image.open(requests.get(URL, stream=True).raw)
    except:
        raw_image = Image.new('RGB', (224, 224), color='red')
        
    pixel_values = processor.image_processor(raw_image, return_tensors="pt").pixel_values.to(device)
    
    q_ids = processor.tokenizer("Caption: What is this?", return_tensors="pt").input_ids.to(device)
    a_car_ids = processor.tokenizer("A car", return_tensors="pt").input_ids.to(device)
    a_noise_ids = processor.tokenizer("Noise", return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        pred_emb = model(pixel_values, q_ids) # [1, Dim]
        target_car = model.y_encoder(a_car_ids)
        target_noise = model.y_encoder(a_noise_ids)
        
        pred_norm = F.normalize(pred_emb, p=2, dim=-1)
        car_norm = F.normalize(target_car, p=2, dim=-1)
        noise_norm = F.normalize(target_noise, p=2, dim=-1)
        
        score_car = (pred_norm @ car_norm.T).item()
        score_noise = (pred_norm @ noise_norm.T).item()
        
    print(f"Similarity to 'A car': {score_car:.4f}")
    print(f"Similarity to 'Noise': {score_noise:.4f}")

def main():
    model_id = "google/paligemma-3b-mix-224"
    print(f"Loading VLJEPA with {model_id}...")
    model = VLJEPA(model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    train_vljepa(model, lambda: get_overfit_batch(processor), n_epoch=1, lr=1e-4)
    test_retrieval(model, processor)

def cli():
    test_sanity()

if __name__ == '__main__':
    main()
