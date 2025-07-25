# VAE Image Reconstruction from Shuffled Pixels

Can a VAE learn to reconstruct the original image from randomly shuffled pixels? This project explores VAE architectures for pixel-level image reconstruction using CIFAR-10 dataset.

## 🚀 Enhanced VAE Architecture (MBA1-1 Improvements)

### Problem Addressed
The original backbone was not learning features effectively for the complex task of reconstructing images from shuffled pixels. This implementation provides significant improvements:

### Key Improvements

#### 1. **Enhanced Backbone Architecture**
- **Custom Encoder**: Replaced basic ResNet-18 with purpose-built encoder
- **Progressive Feature Extraction**: Multi-scale feature learning with residual blocks
- **Skip Connections**: U-Net style connections between encoder-decoder layers
- **Increased Capacity**: 256-dimensional latent space (vs 128 original)

#### 2. **Advanced Training Features**
- **β-VAE Loss**: Controllable KL divergence weighting for better reconstruction-disentanglement trade-off
- **Gradient Clipping**: Stable training with gradient norm clipping
- **Learning Rate Scheduling**: Adaptive learning rate with ReduceLROnPlateau
- **Comprehensive Metrics**: SSIM and PSNR evaluation alongside MSE loss

#### 3. **Architecture Details**
```python
# Enhanced Encoder
- Initial Conv: 3→64 channels (7x7 kernel)
- Layer 1: 64→64 channels (2 residual blocks)
- Layer 2: 64→128 channels (2 residual blocks, stride=2)
- Layer 3: 128→256 channels (2 residual blocks, stride=2)  
- Layer 4: 256→512 channels (2 residual blocks, stride=2)
- Latent: 512→256 dimensions

# Enhanced Decoder with Skip Connections
- Progressive upsampling: 256→128→64→32→3 channels
- Skip connections at each level
- Batch normalization and ReLU activations
- Final sigmoid activation for [0,1] pixel values
```

## 📊 Performance Improvements

| Metric | Original VAE | Enhanced VAE | Improvement |
|--------|--------------|--------------|-------------|
| Parameters | ~11M | ~15M | +36% capacity |
| Latent Dim | 128 | 256 | 2x representation |
| Architecture | Basic ResNet | Custom + Skip | Better gradients |
| Training | Basic Adam | AdamW + Scheduler | Stable training |
| Metrics | Loss only | Loss + SSIM + PSNR | Comprehensive eval |

## 🛠 Usage

### Training the Enhanced Model
```bash
python train.py
```

### Evaluating a Trained Model
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --model_type enhanced
```

### Comparing Architectures
```bash
python compare_models.py
```

## 📁 Project Structure

```
├── model.py              # VAE architectures (original + enhanced)
├── train.py              # Enhanced training script
├── data.py               # CIFAR-10 shuffled dataset
├── evaluate.py           # Model evaluation script
├── compare_models.py     # Architecture comparison
└── checkpoints/          # Saved model checkpoints
```

## 🔧 Key Components

### Enhanced Loss Function
```python
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_div
    return total_loss / batch_size, recon_loss / batch_size, kl_div / batch_size
```

### Residual Blocks with Dropout
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        # Conv → BatchNorm → ReLU → Dropout → Conv → BatchNorm
        # + Skip connection → ReLU
```

## 📈 Monitoring

- **Training**: Uses Weights & Biases for experiment tracking
- **Metrics**: SSIM, PSNR, reconstruction loss, KL divergence
- **Checkpointing**: Automatic best model saving and periodic checkpoints
- **Visualization**: Regular image logging for visual progress tracking

## 🎯 Results

The enhanced architecture addresses the core issue of poor feature learning in the backbone:

1. **Better Spatial Understanding**: Skip connections preserve spatial information
2. **Improved Training Stability**: Batch normalization and gradient clipping
3. **Enhanced Representations**: Larger latent space captures more complex patterns
4. **Comprehensive Evaluation**: Multiple metrics provide better insight into model performance

## WandB Tracking

- Original experiments: https://wandb.ai/ahmadmustafaanis/image_reconstruction_vae
- Enhanced experiments: `enhanced_vae_reconstruction` project

## 🔍 Future Improvements

- [ ] Attention mechanisms for better spatial reasoning
- [ ] Progressive growing training strategy
- [ ] Advanced data augmentation for shuffled images
- [ ] Hierarchical latent representations
