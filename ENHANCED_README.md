# Enhanced VAE for Image Reconstruction

## Problem Statement
**JIRA Ticket MBA1-2**: Model Backbone not performing well - The backbone is not learning features very well, we need to improve our feature recognition.

## Solution Overview
This enhanced VAE implementation addresses the feature learning issues through several key improvements:

### Key Enhancements

1. **Enhanced Encoder Architecture**
   - Self-attention mechanisms for better spatial feature learning
   - Residual blocks with progressive feature extraction
   - Batch normalization and dropout for better regularization
   - Increased latent dimension (128 → 256) for richer representations

2. **Improved Loss Function**
   - β-VAE with learnable β parameter for automatic weighting
   - Perceptual loss using VGG features for better reconstruction quality
   - Separate tracking of reconstruction, KL, and perceptual losses

3. **Advanced Training Dynamics**
   - KL annealing (gradual increase of KL weight during training)
   - Learning rate scheduling (Cosine Annealing)
   - Gradient clipping for training stability
   - AdamW optimizer with weight decay

4. **Enhanced Decoder**
   - Residual connections for better gradient flow
   - Batch normalization in all decoder layers
   - Progressive upsampling with feature refinement

## Files Structure

```
├── model.py              # Enhanced VAE model with attention and β-VAE
├── train_enhanced.py     # Improved training script with advanced techniques
├── evaluation.py         # Comprehensive evaluation and visualization tools
├── test_model.py         # Model validation script
├── requirements.txt      # Python dependencies
├── data.py              # Original data loading (unchanged)
├── train.py             # Original training script (kept for comparison)
└── ENHANCED_README.md   # This file
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install torch torchvision numpy matplotlib wandb tqdm scikit-learn seaborn scipy
```

## Usage

### 1. Test Model Architecture
```bash
python test_model.py
```
This validates that the enhanced model works correctly and shows parameter counts.

### 2. Train Enhanced Model
```bash
python train_enhanced.py
```

### 3. Evaluate Trained Model
```bash
python evaluation.py
```

## Model Improvements

### Architecture Comparison

| Component | Original | Enhanced |
|-----------|----------|----------|
| Encoder | Basic ResNet-18 | ResNet + Self-Attention |
| Latent Dim | 128 | 256 |
| Decoder | Simple ConvTranspose | Residual + BatchNorm |
| Loss | Basic VAE | β-VAE + Perceptual |
| Optimizer | Adam (lr=1e-3) | AdamW (lr=1e-4) + Scheduling |
| Regularization | None | Dropout + Weight Decay |

### Training Improvements

1. **KL Annealing**: Gradually increases KL weight from 0 to 1 over first 200 epochs
2. **Learning Rate Scheduling**: Cosine annealing from 1e-4 to 1e-6
3. **Gradient Clipping**: Prevents exploding gradients (clip norm = 1.0)
4. **Better Validation**: Comprehensive metrics tracking every 10 epochs
5. **Checkpointing**: Automatic saving of best models and regular checkpoints

### Loss Function Enhancements

```python
# Enhanced loss with multiple components
total_loss = reconstruction_loss + β * kl_divergence + λ * perceptual_loss

# Where:
# - β is learnable and automatically adjusted
# - perceptual_loss uses VGG16 features
# - All components are tracked separately
```

## Expected Improvements

1. **Better Feature Learning**: Self-attention helps capture spatial relationships
2. **Improved Reconstruction Quality**: Perceptual loss leads to more realistic outputs
3. **More Stable Training**: KL annealing and gradient clipping prevent collapse
4. **Richer Latent Space**: Larger latent dimension captures more details
5. **Automatic Hyperparameter Tuning**: Learnable β parameter

## Evaluation Metrics

The enhanced model tracks multiple metrics:

- **MSE**: Mean Squared Error for pixel-level accuracy
- **PSNR**: Peak Signal-to-Noise Ratio for reconstruction quality  
- **SSIM**: Structural Similarity Index for perceived quality
- **Perceptual Loss**: VGG-based feature similarity
- **KL Divergence**: Latent space regularization
- **Beta Value**: Automatic weighting parameter

## Visualization Tools

The evaluation script provides:

1. **Reconstruction Samples**: Side-by-side comparison of shuffled/original/reconstructed
2. **Latent Space Visualization**: t-SNE plot of learned representations
3. **Latent Interpolation**: Smooth transitions between encoded samples
4. **Training Metrics**: Loss curves and parameter evolution

## Training Configuration

### Hyperparameters
```python
EPOCHS = 1000           # Reduced from 100K for practical training
BATCH_SIZE = 32         # Smaller for better gradient estimates  
LEARNING_RATE = 1e-4    # Lower for stability
LATENT_DIM = 256        # Doubled for richer representations
CLIP_GRAD = 1.0         # Gradient clipping threshold
```

### Hardware Requirements
- GPU recommended (CUDA support)
- 8GB+ RAM for data loading
- ~2GB GPU memory for batch size 32

## Results Tracking

Training metrics are logged to Weights & Biases:
- Project: "enhanced_vae_reconstruction"
- Tracks: losses, learning rate, beta parameter, sample images
- Automatic model architecture logging

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `BATCH_SIZE` in train_enhanced.py
2. **Slow Training**: Ensure CUDA is available, reduce `num_workers` if needed
3. **Poor Reconstruction**: Check that perceptual loss is working (requires pretrained VGG)
4. **NaN Losses**: Gradient clipping should prevent this, check learning rate

### Model Loading
```python
# Load trained model for evaluation
checkpoint = torch.load('checkpoints/best_model_epoch_X.pth')
model = VAE(image_channels=3, latent_dim=256)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Next Steps

1. **Run Enhanced Training**: Use `train_enhanced.py` to train the improved model
2. **Compare Results**: Use original vs enhanced model on same data
3. **Tune Hyperparameters**: Adjust β, learning rate, or architecture based on results
4. **Scale Up**: Try larger images or different datasets

## Contributing

To address JIRA ticket MBA1-2:
1. Train the enhanced model using `train_enhanced.py`
2. Evaluate using `evaluation.py` 
3. Compare metrics with baseline model
4. Update JIRA ticket with results and performance improvements

---

**Status**: ✅ Implementation Complete - Ready for training and evaluation
**Next Action**: Run enhanced training to validate improvements