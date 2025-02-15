from sklearn.model_selection import train_test_split
import os
import shutil

dataset_directory = "C:/Users/rhogu/Downloads/dataImage"
train_directory = "C:/Users/rhogu/Downloads/dataImage/train"
val_directory = "C:/Users/rhogu/Downloads/dataImage/val"
test_directory = "C:/Users/rhogu/Downloads/dataImage/test"

# List all image files in the dataset directory
all_files = os.listdir(dataset_directory)

image_extensions = ['.jpg', '.jpeg', '.png']
image_files = [file for file in all_files if os.path.splitext(file)[1].lower() in image_extensions]

# Split the dataset into training, validation, and testing sets
train_files, left_files = train_test_split(image_files, test_size = 0.4, random_state = 42)
val_files, test_files = train_test_split(left_files, test_size = 0.5, random_state = 42)

# Move files to their respective directories
for file in train_files:
    shutil.copy(os.path.join(dataset_directory, file), os.path.join(train_directory, file))

for file in val_files:
    shutil.copy(os.path.join(dataset_directory, file), os.path.join(val_directory, file))

for file in test_files:
    shutil.copy(os.path.join(dataset_directory, file), os.path.join(test_directory, file))
