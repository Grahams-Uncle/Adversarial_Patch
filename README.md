# Adversarial_Patch

Pytorch implementation of Adversarial Patch[1] on CIFAR-10 dataset. Patch generation algorithm design is inspired by adversarial patch implementation in [2][3].

## Adversarial Patch
We generated adversarial patches of different sizes using targeted and untargeted attack strategies. Each patch is optimized across 5000 randomly selected images from the CIFAR10 train dataset, with this process being repeated for each epoch over a total of 5 epochs.\
\
![](https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/patch_img/patch_collection.png?raw=true)

## Experiment Results
We selected pre-trained ResNet50, VGG16, and DenseNet for adversarial patch generation and evaluation. Pre-trained ResNet50 model is used for white-box attacks, while the pre-trained VGG16 and DenseNet are used for black-box attacks. \
\
VGG16 model prediction on unaltered image:\
<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/clean_bird.png" width="400">\
VGG16 model prediction on the patched image (with 'dog' as target):\
<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/adv_bird_with_dog.png" width="400">

The Adversarial Success Rate for untargeted attacks was determined by dividing the number of instances where the model incorrectly predicted the patched image, but was correct on the unpatched image, by the total count of correctly predicted unpatched images. The Adversarial Success Rate for targeted attacks was calculated as the ratio of instances where its prediction changed to the adversarial target class to the number of correctly predicted instances (excluding the adversarial target class).  The range of patch sizes tested varied from 8x8 to 64x64.\
\
Attack Success Rate vs Patch Size:\
<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/ASR_1.png" height="300">
<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/ASR_2.png" height="300">

 ## Reference
 [1] [Adversarial Patch](https://arxiv.org/abs/1712.09665)\
 [2] [Adversarial Attacks Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial10/Adversarial_Attacks.html)\
 [3] [Adversarial Patch Attack Repository](https://github.com/A-LinCui/Adversarial_Patch_Attack)

