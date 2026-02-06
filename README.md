# Biomedical VL-JEPA (PyTorch)

A PyTorch implementation of **VL-JEPA** specialized for biomedical domains, using **BioBERT** and a TinyViT (POW) for now need to replace with **VMamba** Backbone.

## Architecture
- **Text Encoder**: Frozen `dmis-lab/biobert-base-cased-v1.1`.
- **Vision Encoder**: TinyViT for now need to replace with VMamba Backbone.
- **Predictor**: Frozen BioBERT with **DoRA** adapters (Rank 16).

## Training Logic
Uses Joint Embedding Predictive Architecture (JEPA) with a **Dual Loss**:
1.  **Reconstruction (MSE)**: Predictor reconstructs latent features of masked image patches (~70% masking).
2.  **Global Alignment (InfoNCE)**: Aligns global vision features with BioBERT text embeddings.

## Usage
```bash
pip install -e .
python model/main.py
```
