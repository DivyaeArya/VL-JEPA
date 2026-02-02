# VL-JEPA (PyTorch)

PyTorch reimplementation of **VL-JEPA**, translated from the original MLX code:  
https://github.com/JosefAlbers/VL-JEPA/tree/main

Based on the **VL-JEPA paper** and **JEPA (LeCun et al.)** ideas.  
Uses **PaLI-Gemma** as the vision–language backbone.

### Extras added
- [x] PyTorch version (from MLX)
- [x] PaLI-Gemma integration
- [x] JEPA-style masking
  - [x] masked context encoder
  - [x] unmasked target encoder
  - [x] stop-gradient on targets
