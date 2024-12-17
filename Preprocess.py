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

    def __init__(self, raw_dir, process_dir, image_size, delete_process_dir=False, batch_size=1000, aug_amount=0):
        """
        Initializes the Process class.

        :param raw_dir: Path to the directory containing raw images.
        :param process_dir: Path where processed images will be saved.
        :param image_size: Desired size for the images.
        :param delete_process_dir: Whether to delete the existing processed directory before creating new one.
        :param batch_size: Number of images to process at once.
        :param aug_amount: Number of augmentations per image (only for training).
        """
        self.RAW_DIR = raw_dir
        self.PROCESSED_DIR = process_dir
        self.LABELS = ["fake", "real"]
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size
        self.AUG_AMOUNT = aug_amount

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

        :param train_ratio: Proportion of data for training.
        :param val_ratio: Proportion of data for validation.
        :param fraction: Fraction of data to process (default is 1 for full dataset).
        """
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        for category in self.LABELS:
            self.process_label(train_ratio, val_ratio, category, self.IMAGE_SIZE, fraction)

    def process_label(self, train_ratio, val_ratio, label, size, fraction):
        """
        Processes the images for one label (fake/real), splits the dataset into
        train/validation/test, and applies transformations.

        :param train_ratio: Proportion of data for training.
        :param val_ratio: Proportion of data for validation.
        :param label: Label/category of the images (fake/real).
        :param size: Desired size for the images.
        :param fraction: Fraction of data to process.
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

        :param base_path: Directory path to save the image.
        :param base_name: Base name for the image.
        :param counter: Counter to ensure uniqueness.
        :return: A unique filename for the image.
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

        :param images: List of image file paths to process.
        :param category: Category for the images (train/valid/test).
        :param size: Desired size for the images.
        :param label: Label/category of the images (fake/real).
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
