import unittest
import os
import shutil
from Preprocess import Process
from PIL import Image

class TestProcess(unittest.TestCase):
    def setUp(self):
        # Set up paths for testing
        self.raw_dir = 'test_raw'
        self.processed_dir = 'test_processed'
        self.image_size = (64, 64)
        self.batch_size = 10
        self.aug_amount = 2

        # Create a mock "raw" directory with "real" and "fake" images
        os.makedirs(os.path.join(self.raw_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(self.raw_dir, 'fake'), exist_ok=True)
        
        # Create dummy image files (valid images for testing purposes)
        for category in ['real', 'fake']:
            for i in range(1, 3):  # Creating two images per category
                image_path = os.path.join(self.raw_dir, category, f'img{i}.jpg')
                img = Image.new('RGB', self.image_size, color=(255, 0, 0))  # Red image for testing
                img.save(image_path)

    def tearDown(self):
        # Clean up test directories
        if os.path.exists(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)

    def test_process_label(self):
        # Test if the directories are created and images are processed
        processor = Process(
            raw_dir=self.raw_dir,
            process_dir=self.processed_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            aug_amount=self.aug_amount
        )
        
        # Call the process method for "real" images (test processing)
        processor.process_label(train_ratio=0.8, val_ratio=0.1, label='real', size=self.image_size, fraction=1)
        
        # Verify if the directories have been created
        train_dir = os.path.join(self.processed_dir, f"{self.image_size[0]}/train/rgb")
        self.assertTrue(os.path.exists(train_dir), "Train directory should be created.")
        
        # Verify if at least one image was processed (based on your logic, this may need adjustment)
        self.assertGreater(len(os.listdir(train_dir)), 0, "Train directory should contain processed images.")

    def test_create_images(self):
        # Test if images are being created properly in the processed folder
        processor = Process(
            raw_dir=self.raw_dir,
            process_dir=self.processed_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            aug_amount=self.aug_amount
        )

        raw_images = [
            os.path.join(self.raw_dir, 'real', 'img1.jpg'),
            os.path.join(self.raw_dir, 'fake', 'img1.jpg')
        ]
        processor.create_images(raw_images, 'train', self.image_size, 'real')
        
        # Check if images were processed and saved
        processed_dir = os.path.join(self.processed_dir, f"{self.image_size[0]}/train/rgb")
        self.assertGreater(len(os.listdir(processed_dir)), 0, "Processed images should be created.")

    def test_augmentation(self):
        # Test if augmentation is applied
        processor = Process(
            raw_dir=self.raw_dir,
            process_dir=self.processed_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            aug_amount=1  # Apply 1 augmentation per image
        )

        raw_images = [
            os.path.join(self.raw_dir, 'real', 'img1.jpg')
        ]
        processor.create_images(raw_images, 'train', self.image_size, 'real')

        # Check if augmented images are created
        processed_dir = os.path.join(self.processed_dir, f"{self.image_size[0]}/train/rgb")
        augmented_images = [f for f in os.listdir(processed_dir) if f.startswith('real_')]
        
        self.assertGreater(len(augmented_images), 1, "Augmentation should create additional images.")

if __name__ == '__main__':
    unittest.main()
