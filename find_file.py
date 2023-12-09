import os
import fnmatch

def find_files(directory, pattern):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            matching_files.append(os.path.join(root, filename))
    return matching_files

def main():
    directory = '/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/results/central_nocrop/test_latest/images'
    pattern = '*fake*'
    
    matching_files = find_files(directory, pattern)

    if matching_files:
        print("Matching files:")
        for file in matching_files:
            print(file)
    else:
        print("No matching files found.")

if __name__ == "__main__":
    main()
    