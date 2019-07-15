import numpy as np
from os import path

dataset_path = "../Data/"
dataset_out = "../Data_split/"

X_train_1 = np.load(path.join(dataset_path, "train-input_1.npy"), mmap_mode='r')
y_train_1 = np.load(path.join(dataset_path, "train-output_1.npy"), mmap_mode='r')

X_train_2 = np.load(path.join(dataset_path, "train-input_2.npy"), mmap_mode='r')
y_train_2 = np.load(path.join(dataset_path, "train-output_2.npy"), mmap_mode='r')

X_val_1 = np.load(path.join(dataset_path, "validation-input_1.npy"), mmap_mode='r')
y_val_1 = np.load(path.join(dataset_path, "validation-output_1.npy"), mmap_mode='r')

X_val_2 = np.load(path.join(dataset_path, "validation-input_2.npy"), mmap_mode='r')
y_val_2 = np.load(path.join(dataset_path, "validation-output_2.npy"), mmap_mode='r')

X_train_1_splits = np.split(X_train_1, 4)
for i, split in enumerate(X_train_1_splits):
    np.save(path.join(dataset_out, "train-input_1-"+str(i)+".npy"), split)
    i+1

y_train_1_splits = np.split(y_train_1, 4)
for i, split in enumerate(y_train_1_splits):
    np.save(path.join(dataset_out, "train-output_1-"+str(i)+".npy"), split)
    i+1

X_train_2_splits = np.split(X_train_2, 4)
for i, split in enumerate(X_train_2_splits):
    np.save(path.join(dataset_out, "train-input_2-"+str(i)+".npy"), split)
    i+1

y_train_2_splits = np.split(y_train_2, 4)
for i, split in enumerate(y_train_2_splits):
    np.save(path.join(dataset_out, "train-output_2-"+str(i)+".npy"), split)
    i+1

X_test_1_split = np.split(X_val_1, 4)
for i, split in enumerate(X_test_1_split):
    np.save(path.join(dataset_out, "validation-input_1-"+str(i)+".npy"), split)
    i+1

y_test_1_split = np.split(y_val_1, 4)
for i, split in enumerate(y_test_1_split):
    np.save(path.join(dataset_out, "validation-output_1-"+str(i)+".npy"), split)
    i+1

X_test_2_split = np.split(X_val_2, 4)
for i, split in enumerate(X_test_2_split):
    np.save(path.join(dataset_out, "validation-input_2-"+str(i)+".npy"), split)
    i+1

y_test_2_split = np.split(y_val_2, 4)
for i, split in enumerate(y_test_2_split):
    np.save(path.join(dataset_out, "validation-output_2-" + str(i) + ".npy"), split)
    i + 1
