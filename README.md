# SinGAN on Swift for TensorFlow

- arXiv: [SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/abs/1905.01164)
- [Supplementary Material](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Shaham_SinGAN_Learning_a_ICCV_2019_supplemental.pdf)
- Offiicial implementation: [tamarott/SinGAN](https://github.com/tamarott/SinGAN)

## Differences from original

### Instance norm instead of batch norm

Original implementation uses batch norm. I afraid it's problematic.  
SinGAN is trained with single image. It means batch size is always 1.  
Therefore batch norm works like instance norm while training.  
But when it comes to inference phase, batch norm uses running stats of training phase. It can be much different from training phase.  

To avoid this, I simply replaced batch norm with instance norm.


### Cease WGAN-GP training

As I wrote in [the issue](https://github.com/tamarott/SinGAN/issues/59), original implementation of gradient penalty looks wrong.  
Anyway S4TF doesn't support higher-order differentiaion for now. So I decided not to use WGAN-GP.

### Use spectral normalization

Since I didn't use WGAN-GP, I need other techniques to stabilize training.  
I employed [spectral normalization](https://arxiv.org/abs/1802.05957) and use hinge loss.
