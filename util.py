from torchvision.transforms.functional import affine
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torchvision import transforms

# Evaluate model accuracy
def test_model(mdl, loader, device):
    mdl.eval()
    running_correct = 0.
    running_loss = 0.
    running_total = 0.
    with torch.no_grad():
        for batch_idx,(data,labels) in enumerate(loader):
            data = data.to(device); labels = labels.to(device)
            clean_outputs = mdl(data)
            clean_loss = F.cross_entropy(clean_outputs, labels)
            _,clean_preds = clean_outputs.max(1)
            running_correct += clean_preds.eq(labels).sum().item()
            running_loss += clean_loss.item()
            running_total += labels.size(0)
    clean_acc = running_correct/running_total
    clean_loss = running_loss/len(loader)
    mdl.train()
    return clean_acc,clean_loss

# Apply the adversarial patch on the target image at the given location
def apply_patch(image, patch, patch_size, location=(0, 0)):
    for i in range(image.shape[0]):
        image[i, :, location[0]:location[0]+patch_size, location[1]:location[1]+patch_size] = patch
    return image

# Make random transformation of the patch during training 
def random_transform(patch):
    angle = random.uniform(-22, 22)  # Rotation angle in degrees
    translate = [random.uniform(-0.1 * patch.size(-2), 0.1 * patch.size(-2)), 
                 random.uniform(-0.1 * patch.size(-1), 0.1 * patch.size(-1))]  # Translation
    scale = random.uniform(0.8, 1.2)  # Scaling
    shear = random.uniform(-10, 10)  # Shear

    # Apply affine transformation
    # Ensure that this operation creates a new tensor
    transformed_patch = affine(patch.clone(), angle=angle, translate=translate, scale=scale, shear=shear)
    return transformed_patch

# Get a image from a given class
def get_target_image(target_class, target_size, dataloader, dataset):
    for image, label in dataloader:
        if label.item() == target_class:
            image = image.squeeze(0)
            print(image.shape)
            target_image = image
            transform = transforms.Compose([transforms.Resize(target_size)])
            target_image = transform(image)

            plt.figure(figsize=(6, 6))
            plt.imshow(transforms.ToPILImage()(target_image))
            plt.axis('off')
            plt.title(f'Target Image (Class {target_class}):{dataset.classes[target_class]}')
            plt.show()

            break
    return target_image