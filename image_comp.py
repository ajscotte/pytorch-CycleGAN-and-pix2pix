import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_images(image1_path, image2_path):
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Check if the images have the same dimensions
    if img1.shape != img2.shape:
        print("Images have different dimensions. Cannot compare.")
        return

    # Calculate Mean Squared Error (MSE)
    mse = np.sum((img1 - img2) ** 2) / float(img1.size)

    return mse

if __name__ == "__main__":
    # Replace 'image1.jpg' and 'image2.jpg' with the paths to your images
    image1_path = 'image1.jpg'
    image2_path = 'image2.jpg'

    mse = compare_images(image1_path, image2_path)

    if mse is not None:
        print(f"Mean Squared Error: {mse}")
