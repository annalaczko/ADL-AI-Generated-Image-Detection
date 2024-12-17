import os
import shutil
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import cv2

class Process():
    def __init__(self,raw_dir, process_dir, image_size, delete_process_dir=False, batch_size=1000, aug_amount=0):
        self.RAW_DIR=raw_dir
        self.PROCESSED_DIR=process_dir
        self.LABELS = ["fake", "real"]
        self.IMAGE_SIZE=image_size
        self.BATCH_SIZE=batch_size
        self.AUG_AMOUNT = aug_amount

        #we delete the original processed_dir to make the new dataset
        if delete_process_dir and os.path.exists(self.PROCESSED_DIR):
            shutil.rmtree(self.PROCESSED_DIR )

        #transformations for the pictures
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

    #this is the intended function thats available from outside
    def run(self, train_ratio, val_ratio, fraction=1):
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        for category in self.LABELS:
            self.process_label( train_ratio, val_ratio, category, self.IMAGE_SIZE, fraction)

    #processes images from one label (true/false)
    def process_label(self, train_ratio, val_ratio, label, size, fraction):
        print(f"Working on {label}")

        #get directory for label
        raw_path = os.path.join(self.RAW_DIR, label)

        #getting images
        image_paths = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith((".png", ".jpg", ".jpeg"))]

        if len(image_paths) == 0:
            print(f"No images found in {raw_path}")
            return
            
        random.shuffle(image_paths)

        #fraction exception handling, and then fractioning
        if fraction>1:
            fraction=1
        elif fraction<0:
            fraction=0

        fraction_split_index = int(len(image_paths) * fraction)
        image_paths = image_paths[:fraction_split_index]
        
        #splitting
        train_split_index = int(len(image_paths) * train_ratio)
        val_split_index = int(len(image_paths) * (train_ratio + val_ratio))
    
        train_paths = image_paths[:train_split_index]
        val_paths = image_paths[train_split_index:val_split_index]
        test_paths = image_paths[val_split_index:]
    
        self.create_images(train_paths, "train", size, label)
        self.create_images(val_paths, "valid", size, label)

        if (train_ratio+val_ratio<1):
            self.create_images(test_paths, "test", size, label)

    #this function is for finding a unique filename for the images
    def get_unique_filename(self, base_path, base_name, counter=0):
        while True:
            image_name = f"{base_name}_{counter}.jpg"
            filename = os.path.join(base_path, image_name)
            
            if not os.path.exists(filename):
                return image_name 
            counter += 1

    def create_images(self,images, category, size, label):
    
        ROOT_DIR=os.path.join(self.PROCESSED_DIR, f"{size[0]}")

        #creating folders to the appropriate category
        if (category=="train"):
            rgb_path=os.path.join(ROOT_DIR, "train/rgb")
            edge_path=os.path.join(ROOT_DIR, "train/edge")
            sharpen_path=os.path.join(ROOT_DIR, "train/sharpen")
        elif(category=="valid"):
            rgb_path=os.path.join(ROOT_DIR, "valid/rgb")
            edge_path=os.path.join(ROOT_DIR, "valid/edge")
            sharpen_path=os.path.join(ROOT_DIR, "valid/sharpen")
        else:
            rgb_path=os.path.join(ROOT_DIR, "test/rgb")
            edge_path=os.path.join(ROOT_DIR, "test/edge")
            sharpen_path=os.path.join(ROOT_DIR, "test/sharpen")
    
        os.makedirs(rgb_path, exist_ok=True)
    
        image_counter=0

        #processing images in batches
        for i in tqdm(range(0, len(images), self.BATCH_SIZE), desc=f"Processing {category}"): #len(images)
            batch = images[i:i + self.BATCH_SIZE]
            for img_path in batch:
                img = Image.open(img_path).convert("RGB")
    
                resized_img_name = self.get_unique_filename(rgb_path, label, image_counter)
    
                #save a simple resized version
                resized_img = self.resize_transform(img)

                #making the filtered version of the image
                self.make_filters(resized_img, resized_img_name , edge_path, sharpen_path)

                #transform picture back to PIL and save
                resized_img = self.save_transform(resized_img)
                resized_img_path=os.path.join(rgb_path, resized_img_name)
                resized_img.save(resized_img_path)
 
                image_counter += 1

                #in case of the category train, we have the option to augment the image to increase the training set
                if (category=="train"):
                    for aug_idx in range(self.AUG_AMOUNT):
                        aug_img_name= self.get_unique_filename(rgb_path, label, image_counter)

                        #augmentation
                        aug_img = self.augmentation_transforms(img) 

                        #making the filtered version of the image
                        self.make_filters(aug_img, aug_img_name , edge_path, sharpen_path)

                        #transform picture back to PIL and save
                        aug_img = self.save_transform(aug_img)
                        aug_img_path=os.path.join(rgb_path, aug_img_name)
                        aug_img.save(aug_img_path)
        
                        image_counter += 1
    
    # this creates the two filtered image
    def make_filters(self, tensor_image, filename, edge_path, sharpen_path):
    
        image = tensor_image.permute(1, 2, 0).numpy()
    
        os.makedirs(edge_path, exist_ok=True)
        os.makedirs(sharpen_path, exist_ok=True)
        
        kernel_edge = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])
    
    
        kernel_sharpen = np.array([ [0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
        
        #apply edge filter and save picture
        edge_result = cv2.filter2D(image, -1, kernel_edge)
    
        edge_output_path = os.path.join(edge_path, filename)
        cv2.imwrite(edge_output_path, edge_result)
    
        #apply sharpen filter and save picture
        sharpen_result = cv2.filter2D(image, -1, kernel_sharpen)
        
        sharpen_output_path = os.path.join(sharpen_path, filename)
        cv2.imwrite(sharpen_output_path, sharpen_result)           