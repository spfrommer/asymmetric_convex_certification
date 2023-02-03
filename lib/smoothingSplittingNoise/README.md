# Improved, Deterministic Smoothing for L_1 Certified Robustness
Code for the paper "Improved, Deterministic Smoothing for L_1 Certified Robustness" (Preprint: https://arxiv.org/abs/2103.10834) by Alexander Levine and Soheil Feizi. 

Most of the code is in the src directory, which is a fork of the code for (Yang et al. 2020) available at https://github.com/tonyduan/rs4a. Instructions for installing dependencies are reproduced here:

```
conda install numpy matplotlib pandas seaborn 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install torchnet tqdm statsmodels dfply
```

 We have added options for SSN and DSSN smoothing distributions/certificates.

To train (a WideResNet-40 on CIFAR-10) using Independent SSN, 

```
python3 -m src.train  --noise=SplitMethod --sigma=${SIGMA}  --experiment-name=cifar_split_${SIGMA} --output-dir checkpoints
```

To train using DSSN,

```
python3 -m src.train  --noise=SplitMethodDerandomized --seed=0 --sigma=${SIGMA}  --experiment-name=cifar_split_derandomized_${SIGMA} --output-dir checkpoints 
```

To generate Independent SSN certificates,

```
python3 -m src.test --output-dir=checkpoints --noise=SplitMethod --sigma ${SIGMA} --experiment-name=cifar_split_${SIGMA} --noise-batch-size=128
```

To generate DSSN certificates,

```
python3 -m src.test_derandomized --seed 0 --output-dir=checkpoints --noise=SplitMethodDerandomized --sigma ${SIGMA} --experiment-name=cifar_split_derandomized_${SIGMA} --noise-batch-size=128
```

Note that for ImageNet training and testing, the directories where ImageNet files are expected are set using the bash variables $IMAGENET_TRAIN_DIR and $IMAGENET_TEST_DIR. 

Pretrained models and certificate data for CIFAR-10 are included in checkpoints directory. Certificate data for ImageNet is also included here (although checkpoints are not, due to GitHub space constraints.)


++++++++++++++++++++++++++++++++++++

For the denoiser experiments in Appendix D, we used the denoising implementation from (Salman et al. 2020) available at https://github.com/microsoft/denoised-smoothing. This package is forked in the "denoised-smoothing" directory. We have added options for training under uniform  SSN, and DSSN noise. Specifically, denoised-smoothing/train_denoiser.py now takes a --noise_type flag, which can be set to 'uniform', 'split' (Independent SSN) or 'split_derandomized' (DSSN), as well as a --seed flag (for DSSN). Additionally the 'cifar_wrn40' classifier architecture has been added. We trained the 'clean' base classifier as a WideResNet-40 using otherwise standard arguments for this package (as described in Salman et al. 2020, Section A.2), and likewise trained denoisers using standard arguments.

To use a denoiser with the certification framework, there are two workflows. If the denoiser is being used with an unmodified network for clean classification (Stability denoiser or MSE denioser without retraining), the pattern is:

```
python3 -m src.test --output-dir=denoise --noise=SplitMethod --sigma 0.5 --experiment-name=cifar_split_0.5 --model WideResNetCompat --noise-batch-size=128 --save-path <Path to clean classifier model> --denoiser-path <Path to denoiser>
```

(Note that we need to use `WideResNetCompat` to handle differences between the two packages)

If we are training the classifier from scratch on an existing denoiser, the pattern is:

```
python3 -m src.train --noise=SplitMethod --sigma=2.0 --experiment-name=cifar_split_2.0 --output-dir denoise_2 --denoiser-path <Path to denoiser>

python3 -m src.test --dataset-skip 20 --output-dir=denoise_2 --noise=SplitMethod --sigma 2.0 --experiment-name=cifar_split_2.0 --noise-batch-size=128 --with_denoiser
```
