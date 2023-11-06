import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import random
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import matplotlib.image as mpimg
import numpy as np

def plot_grid_images(training_set, class_names, mean, std, rows=3, columns=3, size=14):
    batch_size = rows * columns
    sampler = RandomSampler(training_set, num_samples=batch_size, replacement=True)
    train_loader = DataLoader(training_set, sampler=sampler, shuffle=False, batch_size=batch_size, num_workers=0)
    
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    plt.figure(figsize=(size,size))
    for i in range(rows*columns):
        plt.subplot(rows, columns, i+1)
        plt.title(class_names[labels.numpy()[i]])
        img = images[i].permute(1,2,0)
        img = torch.tensor(std)*img + torch.tensor(mean)
        plt.axis('off')
        plt.imshow(img, interpolation='none')
        plt.tight_layout()
        
def plot_images_per_class(images_path):
    data_folder = images_path
    item_dict = {root.split('/')[-1]: len(files) for root, _, files in os.walk(data_folder)}
   
    plt.figure(figsize=(20,8))
    plt.bar(list(item_dict.keys())[1:], list(item_dict.values())[1:], color='g')
    plt.xticks(rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return e_x / e_x.sum(axis=0)

def display_wrong_sample(result_dict, class_names, datapaths, std=[0.485, 0.456, 0.406], mean=[0.229, 0.224, 0.225]):
    for model_name, (_, _, wrong_sample) in result_dict.items():
        print(f"Wrong sample of {model_name}")
        display_samples = random.sample(wrong_sample, 3)
        for sample in display_samples:
            image, pred, target = sample
            pred = softmax(pred)
            class_pred = pred.argmax(0)

            fig, axs = plt.subplots(2, 3, figsize=(20, 8))

            # Display the image
            image = torch.tensor(std)*image.permute(1, 2, 0) + torch.tensor(mean)

            axs[0, 0].imshow(image)
            axs[0, 0].set_title("Sample")

            # Display the target class image
            target_class = random.sample(glob(f"{datapaths}/{class_names[target]}/*"), 2)
            axs[0, 1].imshow(mpimg.imread(target_class[0]))
            axs[0, 1].set_title(f"Class Target {class_names[target][10:]} with confidence {pred[target]:.2f}")
            axs[1, 1].imshow(mpimg.imread(target_class[1]))
            # axs[2].set_title(f"Class Target {class_names[target][10:]}")

            # Display the predicted class image
            pred_class = random.sample(glob(f"{datapaths}/{class_names[class_pred]}/*"), 2)
            axs[0, 2].imshow(mpimg.imread(pred_class[0]))
            axs[0, 2].set_title(f"Class Predict {class_names[class_pred][10:]} with confidence {pred[class_pred]:.2f}")
            axs[1, 2].imshow(mpimg.imread(pred_class[1]))
            # axs[4].set_title(f"Class Predict {class_names[class_pred][10:]}")
            for ax in axs.ravel():
              ax.axis('off')
            plt.tight_layout()
            plt.show()
        print("="*20)
