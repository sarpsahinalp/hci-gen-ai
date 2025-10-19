import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

mnist = datasets.MNIST(root=".", train=True, download=True, transform=transforms.ToTensor())
X = mnist.data.numpy() / 255.0
y = mnist.targets.numpy()

X_bin = (X > 0.5).astype(np.uint8)

p_pixel_prob = X_bin.mean(axis=0)

p_pixel_prob_class = np.zeros((10, 28, 28))
for i in range(10):
    class_image = X_bin[y == i]
    p_pixel_prob_class[i] = class_image.mean(axis=0)

if _name_ == "_main_":
    choice = int(input("Select a mode\n1. For pure randomness\n2. For classification of numbers\nEnter your choice: "))
    while True:
        number = int(input("Please enter a number you want to generate: "))

        if number == -1:
            choice = int(
                input("Select a mode\n1. For pure randomness\n2. For classification of numbers\nEnter your choice: "))

        if choice == 1:
            rand_image = (np.random.rand(784) < p_pixel_prob.flatten()).astype(int)
            image_2d = rand_image.reshape(28, 28)
        else:
            prob = p_pixel_prob_class[number]
            image_2d = (np.random.rand(28, 28) < prob).astype(int)

        plt.imshow(image_2d, cmap='gray')
        plt.show()