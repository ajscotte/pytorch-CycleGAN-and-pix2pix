# import os
# import fnmatch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def find_files(directory, pattern):
#     matching_files = []
#     for root, dirs, files in os.walk(directory):
#         for filename in fnmatch.filter(files, pattern):
#             matching_files.append(os.path.join(root, filename))
#     return matching_files

# def compare_images(image1_path, image2_path):
#     # Read images
#     img1 = cv2.imread(image1_path)
#     img2 = cv2.imread(image2_path)

#     # Check if the images have the same dimensions
#     if img1.shape != img2.shape:
#         print("Images have different dimensions. Cannot compare.")
#         return

#     # Calculate Mean Squared Error (MSE)
#     mse = np.sum((img1 - img2) ** 2) / float(img1.size)

#     return mse

# def main():
#     directory = '/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/results/central_nocrop/test_latest/images'
#     pattern = '*fake*'
    
#     matching_files = find_files(directory, pattern)

#     if matching_files:
#         print("Matching files:")
#         for file in matching_files:
#             print(file)
#     else:
#         print("No matching files found.")

# if __name__ == "__main__":
#     main()
    
import os
import fnmatch

def find_files(directory, pattern):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            matching_files.append(filename)
    return matching_files

def main():
    directory_central = '/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/results/central_nocrop/test_latest/images'
    pattern = '*fake*'
    
    matching_files = find_files(directory_central, pattern)

    if matching_files:
        print("Matching files:")
        for file in matching_files:
            print(file)
            path_with_filename = os.path.join(directory_central, file)
            print("Path with filename:", path_with_filename)
    else:
        print("No matching files found.")

if __name__ == "__main__":
    main()