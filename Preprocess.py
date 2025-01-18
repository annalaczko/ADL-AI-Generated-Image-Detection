import os
import shutil
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import cv2

class Process():
    """
    Class for processing raw image data by applying transformations, augmentations, 
    and saving processed images in different directories.
    """

    def __init__(self, raw_dir, process_dir, image_size, delete_process_dir=False, batch_size=1000, aug_amount=0, filters=False):
        """
        Initializes the Process class.

        Parameters:
            raw_dir (str): Path to the directory containing raw images.
            process_dir (str): Path where processed images will be saved.
            image_size (tuple): Desired size for the images.
            delete_process_dir (bool): Whether to delete the existing processed directory before creating new one.
            batch_size (int): Number of images to process at once.
            aug_amount (int): Number of augmentations per image (only for training).
        """
        self.RAW_DIR = raw_dir
        self.PROCESSED_DIR = process_dir
        self.LABELS = ["fake", "real"]
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size
        self.AUG_AMOUNT = aug_amount
        self.FILTERS = filters

        # Deleting the original processed_dir to make the new dataset
        if delete_process_dir and os.path.exists(self.PROCESSED_DIR):
            shutil.rmtree(self.PROCESSED_DIR)

        # Defining transformations for the images
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        self.augmentation_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.IMAGE_SIZE, scale=(0.7, 0.9)),
            transforms.RandomApply([
                transforms.ColorJitter(contrast=0.5)
            ], p=0.5),
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        self.save_transform = transforms.ToPILImage()

    def run(self, train_ratio, val_ratio, fraction=1):
        """
        Runs the image processing pipeline: processes images by categories, applies transformations, 
        and creates dataset splits for training, validation, and testing.

        Parameters:
            train_ratio (float): Proportion of data for training.
            val_ratio (float): Proportion of data for validation.
            fraction (float): Fraction of data to process (default is 1 for full dataset).
        """
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        for category in self.LABELS:
            self.process_label(train_ratio, val_ratio, category, self.IMAGE_SIZE, fraction)

    def process_label(self, train_ratio, val_ratio, label, size, fraction):
        """
        Processes the images for one label (fake/real), splits the dataset into 
        train/validation/test, and applies transformations.

        Parameters:
            train_ratio (float): Proportion of data for training.
            val_ratio (float): Proportion of data for validation.
            label (str): Label/category of the images (fake/real).
            size (tuple): Desired size for the images.
            fraction (float): Fraction of data to process.
        """
        print(f"Working on {label}")

        # Get directory for label
        raw_path = os.path.join(self.RAW_DIR, label)  
        # Getting images from the directory
        image_paths = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith((".png", ".jpg", ".jpeg"))]
        
        if len(image_paths) == 0:
            print(f"No images found in {raw_path}")
            return

        random.shuffle(image_paths)

        # Handle fraction parameter
        if fraction > 1:
            fraction = 1
        elif fraction < 0:
            fraction = 0

        fraction_split_index = int(len(image_paths) * fraction)
        image_paths = image_paths[:fraction_split_index]

        # Splitting into train/validation/test sets
        train_split_index = int(len(image_paths) * train_ratio)
        val_split_index = int(len(image_paths) * (train_ratio + val_ratio))

        train_paths = image_paths[:train_split_index]
        val_paths = image_paths[train_split_index:val_split_index]
        test_paths = image_paths[val_split_index:]
        
        self.create_images(train_paths, "train", size, label)
        self.create_images(val_paths, "valid", size, label)

        if (train_ratio + val_ratio < 1):
            self.create_images(test_paths, "test", size, label)

    def get_unique_filename(self, base_path, base_name, counter=0):
        """
        Generates a unique filename by appending a counter to the base name.

        Parameters:
            base_path (str): Directory path to save the image.
            base_name (str): Base name for the image.
            counter (int): Counter to ensure uniqueness.

        Returns:
            str: A unique filename for the image.
        """
        while True:
            image_name = f"{base_name}_{counter}.jpg"
            filename = os.path.join(base_path, image_name)

            if not os.path.exists(filename):
                return image_name
            counter += 1

    def create_images(self, images, category, size, label):
        """
        Processes a list of images, applies transformations, and saves them in the appropriate folder.

        Parameters:
            images (list): List of image file paths to process.
            category (str): Category for the images (train/valid/test).
            size (tuple): Desired size for the images.
            label (str): Label/category of the images (fake/real).
        """
        ROOT_DIR = os.path.join(self.PROCESSED_DIR, f"{size[0]}")

        # Creating directories for each category
        if category == "train":
            rgb_path = os.path.join(ROOT_DIR, "train/rgb")
            edge_path = os.path.join(ROOT_DIR, "train/edge")
            sharpen_path = os.path.join(ROOT_DIR, "train/sharpen")
        elif category == "valid":
            rgb_path = os.path.join(ROOT_DIR, "valid/rgb")
            edge_path = os.path.join(ROOT_DIR, "valid/edge")
            sharpen_path = os.path.join(ROOT_DIR, "valid/sharpen")
        else:
            rgb_path = os.path.join(ROOT_DIR, "test/rgb")
            edge_path = os.path.join(ROOT_DIR, "test/edge")
            sharpen_path = os.path.join(ROOT_DIR, "test/sharpen")

        os.makedirs(rgb_path, exist_ok=True)

        image_counter = 0

        # Processing images in batches
        for i in tqdm(range(0, len(images), self.BATCH_SIZE), desc=f"Processing {category}"):
            batch = images[i:i + self.BATCH_SIZE]
            for img_path in batch:
                img = Image.open(img_path).convert("RGB")

                resized_img_name = self.get_unique_filename(rgb_path, label, image_counter)

                # Save a simple resized version
                resized_img = self.resize_transform(img)

                # Apply filters and save
                if (self.FILTERS):
                    self.make_filters(resized_img, resized_img_name, edge_path, sharpen_path)

                # Convert tensor to PIL and save
                resized_img = self.save_transform(resized_img)
                resized_img_path = os.path.join(rgb_path, resized_img_name)
                resized_img.save(resized_img_path)

                image_counter += 1

                # Optionally augment training images
                if category == "train":
                    for aug_idx in range(self.AUG_AMOUNT):
                        aug_img_name = self.get_unique_filename(rgb_path, label, image_counter)

                        # Apply augmentation
                        aug_img = self.augmentation_transforms(img)

                        # Apply filters and save
                        self.make_filters(aug_img, aug_img_name, edge_path, sharpen_path)

                        # Convert tensor to PIL and save
                        aug_img = self.save_transform(aug_img)
                        aug_img_path = os.path.join(rgb_path, aug_img_name)
                        aug_img.save(aug_img_path)

                        image_counter += 1

    def make_filters(self, tensor_image, filename, edge_path, sharpen_path):
        """
        Applies edge detection and sharpening filters to the image, then saves the results.

        Parameters:
            tensor_image (Tensor): The image tensor to process.
            filename (str): The filename for saving the filtered images.
            edge_path (str): Path to save the edge-filtered images.
            sharpen_path (str): Path to save the sharpen-filtered images.
        """
        image = tensor_image.permute(1, 2, 0).numpy()

        os.makedirs(edge_path, exist_ok=True)
        os.makedirs(sharpen_path, exist_ok=True)

        # Edge detection kernel
        kernel_edge = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])

        # Sharpening kernel
        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])

        # Apply edge filter and save result
        edge_result = cv2.filter2D(image, -1, kernel_edge)

        edge_output_path = os.path.join(edge_path, filename)
        cv2.imwrite(edge_output_path, edge_result)

        # Apply sharpen filter and save result
        sharpen_result = cv2.filter2D(image, -1, kernel_sharpen)

        sharpen_output_path = os.path.join(sharpen_path, filename)
        cv2.imwrite(sharpen_output_path, sharpen_result)
