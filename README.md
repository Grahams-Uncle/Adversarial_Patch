# Adversarial Patch

A PyTorch implementation of [Adversarial Patch](https://arxiv.org/abs/1712.09665) attacks on the CIFAR-10 dataset. This project investigates the effectiveness of targeted and untargeted adversarial patches against convolutional neural networks (CNNs), including their transferability across different model architectures.

The patch generation algorithm is inspired by the implementations and tutorials in references [2] and [3].

## Adversarial Patch

We generated adversarial patches of different sizes using both **targeted** and **untargeted** attack strategies:

- **Targeted attacks:** Optimize a patch to induce the model to predict a specific target class.
- **Untargeted attacks:** Optimize a patch to cause misclassification, regardless of the predicted class.

Each patch was optimized using 5,000 randomly selected images from the CIFAR-10 training dataset per epoch, over a total of 5 epochs.

The following figure shows the generated adversarial patches:

![Generated Adversarial Patches](https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/patch_img/patch_collection.png?raw=true)

## Experiment Results

We evaluated adversarial patch effectiveness using three pretrained CNN architectures: **ResNet-50, VGG-16, and DenseNet**.

- **White-box attack:** ResNet-50 was used to generate and evaluate adversarial patches with access to the model's gradients.
- **Black-box attack:** VGG-16 and DenseNet were used to evaluate the transferability of patches generated using ResNet-50.

### Adversarial Examples

The following example demonstrates how an adversarial patch targeting the **dog** class affects VGG-16's prediction on an image of a bird.

**VGG-16 prediction on the original image:**

<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/clean_bird.png" width="400">

**VGG-16 prediction on the patched image (target class: dog):**

<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/adv_bird_with_dog.png" width="400">

### Attack Success Rate

We measured attack effectiveness using the **Attack Success Rate (ASR)**, considering only images correctly classified before patch application.

- **Untargeted ASR:** The proportion of originally correctly classified images that become misclassified after applying the adversarial patch.
- **Targeted ASR:** The proportion of originally correctly classified images (excluding those belonging to the target class) that are classified as the target class after applying the patch.

We evaluated patch sizes ranging from **8×8 to 64×64 pixels** to investigate how patch size influences attack effectiveness.

**Attack Success Rate vs. Patch Size:**

<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/ASR_1.png" height="300">

<img src="https://github.com/Grahams-Uncle/Adversarial_Patch/blob/main/experiment_img/ASR_2.png" height="300">

## References

1. [Adversarial Patch — Brown et al. (2017)](https://arxiv.org/abs/1712.09665)
2. [Adversarial Attacks Tutorial — UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial10/Adversarial_Attacks.html)
3. [Adversarial Patch Attack — GitHub Repository](https://github.com/A-LinCui/Adversarial_Patch_Attack)
