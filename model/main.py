# Pytorch Implementation of VL-JEPA model and training logic
# Biomedical VL-JEPA with BioBERT + TinyViT for now need to replace with VMamba Backbone

import os
import glob
import json
import time
import math
import copy
from types import SimpleNamespace
import requests
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoProcessor, AutoModel, AutoTokenizer, ViTConfig, ViTModel
from PIL import Image

# Configuration
BIO_BERT_ID = "dmis-lab/biobert-base-cased-v1.1"

# Standard ImageNet mean/std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, alpha, scale, dropout):
        output_dims, input_dims = linear.weight.shape
        lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, scale=scale, dropout=dropout, bias=(linear.bias is not None))
        lora_lin.linear = linear 
        with torch.no_grad():
             lora_lin.m.copy_(torch.linalg.norm(linear.weight, dim=1))
        return lora_lin

    def __init__(self, input_dims, output_dims, r, alpha, scale, dropout, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = nn.Parameter(torch.empty(r, input_dims))
        nn.init.uniform_(self.lora_a, -init_scale, init_scale)
        self.lora_b = nn.Parameter(torch.zeros(output_dims, r))
        self.m = nn.Parameter(torch.zeros(output_dims))

    def _get_weight_norm(self):
        w = self.linear.weight
        return w + (self.scale * self.lora_b @ self.lora_a)

    def forward(self, x):
        y = self.linear(x)
        dropout_x = self.dropout(x)
        lora_out = dropout_x @ self.lora_a.t() @ self.lora_b.t()
        z = y + (self.scale * lora_out)
        
        bias = self.linear.bias if self.linear.bias is not None else 0
        if isinstance(bias,  torch.Tensor):
             z_no_bias = z - bias
        else:
             z_no_bias = z
             
        w_adapted = self._get_weight_norm()
        denom = torch.linalg.norm(w_adapted, dim=1) + 1e-6
        factor = (self.m / denom).view(-1)
        z_out = z_no_bias * factor
        
        if isinstance(bias, torch.Tensor):
            z_out = z_out + bias
        return z_out

def to_dora(module, targets=['query', 'value', 'dense'], rank=8, scale=0.1):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if any(t in name for t in targets):
                dora_layer = DoRALinear.from_linear(child, r=rank, alpha=rank, scale=scale, dropout=0.0)
                setattr(module, name, dora_layer)
        else:
            to_dora(child, targets, rank, scale)

class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size=384, num_layers=4, num_heads=6):
        super().__init__()
        self.config = ViTConfig(
            image_size=224, 
            patch_size=16, 
            num_hidden_layers=num_layers, 
            hidden_size=hidden_size, 
            num_attention_heads=num_heads, 
            intermediate_size=hidden_size*4,
            layer_norm_eps=1e-6
        )
        self.model = ViTModel(self.config)
        self.out_dim = hidden_size
        self.num_patches = (224 // 16) ** 2

    def forward(self, pixel_values, mask=None):
        # mask: [B, num_patches] (True = Masked/Remove)
        
        # 1. Get Embeddings (CLS + Patches + Pos)
        # ViTModel.embeddings return (B, L, D) including CLS at index 0
        embeddings = self.model.embeddings(pixel_values)
        
        if mask is not None:
            B, L, D = embeddings.shape
            # mask is for patches only (length L-1)
            # Create full mask with CLS kept (False)
            cls_keep = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_keep, mask], dim=1) # [B, L]
            
            # Select unmasked
            # We assume number of unmasked is constant per batch for stacking
            keep_indices = ~full_mask
            
            # Use masking to filter
            # To utilize fast tensor ops and maintain batch dim, we usually flat select and view
            # But we need to ensure shape (B, N_unmasked, D)
            # Since MaskingGenerator ensures constant ratio, N_unmasked is constant.
            
            embeddings = embeddings[keep_indices].view(B, -1, D)
            
        # 2. Pass to Encoder
        # ViTModel.encoder(embeddings) -> LastHiddenState
        encoded = self.model.encoder(embeddings).last_hidden_state
        return encoded


class BioTextEncoder(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(BIO_BERT_ID)
        self.out_dim = self.model.config.hidden_size 
        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MaskingGenerator:
    def __init__(self, input_size=(224//16, 224//16), mask_ratio=0.7):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.mask_ratio = mask_ratio
        
    def __call__(self, batch_size=1):
        num_mask = int(self.mask_ratio * self.num_patches)
        mask = np.hstack([
            np.zeros((batch_size, self.num_patches - num_mask)),
            np.ones((batch_size, num_mask)),
        ])
        rng = np.random.default_rng()
        for i in range(batch_size):
            rng.shuffle(mask[i])
        return torch.from_numpy(mask).bool() # [B, L], True = Masked

class VLJEPA(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        
        # 1. Vision Encoders (Student & Teacher)
        self.student_vision = TinyVisionEncoder()
        
        self.teacher_vision = copy.deepcopy(self.student_vision)
        for p in self.teacher_vision.parameters():
            p.requires_grad = False
            
        self.vision_projector = ProjectionHead(self.student_vision.out_dim, latent_dim)
        self.teacher_projector = copy.deepcopy(self.vision_projector)
        for p in self.teacher_projector.parameters():
             p.requires_grad = False
             
        # 2. Text Encoder (BioBERT)
        self.text_encoder = BioTextEncoder(frozen=True)
        self.text_projector = ProjectionHead(self.text_encoder.out_dim, latent_dim)
        
        # 3. Predictor (BioBERT based)
        self.predictor_backbone = AutoModel.from_pretrained(BIO_BERT_ID)
        for p in self.predictor_backbone.parameters():
            p.requires_grad = False 
        to_dora(self.predictor_backbone, rank=16) 
        
        self.predictor_projector = nn.Linear(latent_dim, self.predictor_backbone.config.hidden_size)
        self.predictor_out = nn.Linear(self.predictor_backbone.config.hidden_size, latent_dim)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.predictor_backbone.config.hidden_size))
        # 2D Pos embeddings for Predictor (14x14)
        self.pred_pos_embed = nn.Parameter(torch.randn(1, 196, self.predictor_backbone.config.hidden_size))

    def update_teacher(self, decay=0.996):
        with torch.no_grad():
            for tp, sp in zip(self.teacher_vision.parameters(), self.student_vision.parameters()):
                tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)
            for tp, sp in zip(self.teacher_projector.parameters(), self.vision_projector.parameters()):
                tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)

    def forward(self, pixel_values, input_ids, mask=None):
        # mask: [B, num_patches] Boolean (True=Masked)
        # 1. Teacher (Target) - Full Image
        with torch.no_grad():
            target_feat = self.teacher_vision(pixel_values) # [B, 197, 384]
            target_feat = self.teacher_projector(target_feat) # [B, 197, 768]
            target_patches = target_feat[:, 1:, :] # [B, 196, 768]
            
        # 2. Student (Context) - Masked inputs
        # Pass mask to efficiently encode only visible patches
        student_feat = self.student_vision(pixel_values, mask=mask) # [B, N_vis+1, 384]
        student_feat = self.vision_projector(student_feat) # [B, N_vis+1, 768]
        student_patches = student_feat[:, 1:, :] # [B, N_vis, 768]
        
        # 3. Text Features (Global Alignment Targets)
        with torch.no_grad():
            text_full, text_pool = self.text_encoder(input_ids)
            text_feat = self.text_projector(text_pool) # [B, 768]
        
        # 4. Predictor Logic
        # Project student features to predictor dimension (if different, but here same)
        # We need to reconstruct the full sequence for the Predictor (BioBERT)
        # to allow it to use 2D relative attention (via pos embeddings) and predict masked spots.
        
        B, N_vis, D = student_patches.shape
        num_patches = self.student_vision.num_patches
        
        # Initialize with Mask Tokens
        # mask is [B, 196]
        # We want [B, 196, D]
        predictor_input = self.mask_token.expand(B, num_patches, -1).clone()
        
        if mask is not None:
             # Scatter unmasked features into their original positions
             # student_patches corresponds to ~mask positions
             # We need to ensure we scatter correctly per batch item
             # PyTorch boolean indexing on [B, L, D] with mask [B, L] flattens the specific dims
             # predictor_input[~mask] = student_patches.flatten(0, 1) 
             # But student_patches is [B, N_vis, D]. flatten -> [B*N_vis, D]
             # ~mask is [B, 196]. sum is B*N_vis.
             # So this assignment works perfectly.
             
             predictor_input[~mask] = student_patches.reshape(-1, D)
        else:
             # If no mask, student_patches is full (except shape mismatch if variable)
             # But masking generator usually always returns mask.
             pass
             
        # Add Positional Embeddings
        predictor_input = predictor_input + self.pred_pos_embed
        
        # Run Predictor
        # BioBERT expects inputs_embeds. It will add its own pos embeddings? 
        # Standard BERT adds absolute pos embeddings.
        # We are providing inputs_embeds.
        # Ideally we should pass position_ids=None (default 0..N) or our own.
        # Since we added `pred_pos_embed` (learnable), we might be double adding if BERT adds its own.
        # But BioBERT's pos embeddings are 1D (text). We want 2D spatial.
        # Adding ours is fine. BERT's pos embeddings are likely small or we can zero them if we wanted.
        # But typically we just add ours.
        
        pred_out = self.predictor_backbone(inputs_embeds=predictor_input).last_hidden_state 
        pred_latents = self.predictor_out(pred_out) # [B, 196, 768]
        
        return pred_latents, target_patches, text_feat, student_patches.mean(1)

def infonce_loss(pred, target, temp=0.07):
    # Normalized
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    logits = (p @ t.T) / temp
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)

def train_dummy(model, steps=5):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    model.train()
    mask_gen = MaskingGenerator()
    
    # Dummy Data
    B = 2
    img = torch.randn(B, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (B, 20)) #(valid range for bert tokens)
    
    print("Starting Dummy Training Loop...")
    for i in range(steps):
        optimizer.zero_grad()
        
        mask = mask_gen(B).to(img.device) # [B, 196]
        
        pred_latents, target_patches, text_feat, student_global = model(img, input_ids, mask)
        
        # Loss 1: Reconstruction (MSE) on MASKED patches
        # Only compute loss where mask=True
        loss_recon = F.mse_loss(pred_latents[mask], target_patches[mask])
        
        # Loss 2: Global Alignment (InfoNCE)
        # Align Student Global with Text
        loss_align = infonce_loss(student_global, text_feat)
        
        loss = loss_recon + 0.1 * loss_align
        
        loss.backward()
        optimizer.step()
        model.update_teacher()
        
        print(f"Step {i}: Loss {loss.item():.4f} (Recon {loss_recon.item():.4f}, Align {loss_align.item():.4f})")

def main():
    print(f"Initializing VL-JEPA with {BIO_BERT_ID}...")
    model = VLJEPA()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable/1e6:.2f}M Trainable / {total/1e6:.2f}M Total")
    
    train_dummy(model)

if __name__ == '__main__':
    main()
