# Downscaling Fourier Neural Operator

This repository contains the code for the paper ["Fourier Neural Operators for Arbitrary Resolution Climate Data Downscaling"](https://arxiv.org/abs/2305.14452).

The paper presents a novel downscaling Fourier neural operator (DSFNO) that can downscale (i.e. super-resolve) numerical PDE solutions and climate model outputs to an arbitrary resolution unseen during training. It trains with data of a small upsampling factor and then can zero-shot downscale its input to arbitrary unseen high resolution. Evaluated both on ERA5 climate model data and on the Navier-Stokes equation solution data, our downscaling model significantly outperforms state-of-the-art convolutional and generative adversarial downscaling models, both in standard single-resolution downscaling and in zero-shot generalization to higher upsampling factors.

Please cite our work when it inspires your research!

# Code and Data

## Code

This repository contains the code for the DSFNO model applied to Navier-Stokes equation solution data downscaling. It trains a DSFNO model to downscale from 16x16 to 32x32 resolution, then evaluates its performance on 16x16 --> 16x16, 16x16 --> 32x32, and 16x16 --> 64x64 downscaling tasks. 

## Data

The data used in this repository is the Navier-Stokes equation solution data from the paper ["Fourier Neural Operator for Parametric Partial Differential Equations"](https://arxiv.org/abs/2010.08895). The solution data's original resolution is 64x64, then it is downsampled to 32x32 and 16x16 using average pooling.

# References

```
@misc{yang2023fourierneuraloperatorsarbitrary,
      title={Fourier Neural Operators for Arbitrary Resolution Climate Data Downscaling}, 
      author={Qidong Yang and Alex Hernandez-Garcia and Paula Harder and Venkatesh Ramesh and Prasanna Sattegeri and Daniela Szwarcman and Campbell D. Watson and David Rolnick},
      year={2023},
      eprint={2305.14452},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.14452}, 
}
```