import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from unittest.mock import MagicMock
import numpy as np
import torchvision

from NeuralNetworkDIY import ClassifierNetwork, Model

class DummyDataset(Dataset):
    """
    A dummy dataset class for testing purposes, simulating images and labels.

    Attributes:
        num_samples (int): Number of samples in the dataset.
        image_size (tuple): Size of the images (height, width).
    """
    def __init__(self, num_samples=10, image_size=(32, 32)):
        """
        Initializes the DummyDataset with specified number of samples and image size.

        :param num_samples: Number of samples in the dataset (default is 10).
        :param image_size: Size of each image as a tuple (height, width) (default is (32, 32)).
        """
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: The number of samples (num_samples).
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a random image, its label, and its path for a given index.

        :param idx: Index of the sample.
        :return: A tuple of (image, label, path).
        """
        image = torch.randn(3, *self.image_size)  # Random tensor simulating an image
        label = torch.randint(0, 2, (1,)).item()  # Random binary label
        path = f"image_{idx}.jpg"  # Dummy path
        return image, label, path


class TestClassifierNetwork(unittest.TestCase):
    """
    Unit tests for the ClassifierNetwork class.

    Tests include:
        - Forward pass functionality.
        - Computation of the flattened size for the input images.
    """
    def setUp(self):
        """
        Sets up the test environment by initializing a ClassifierNetwork instance
        with a specified image size.
        """
        self.image_size = (32, 32)
        self.network = ClassifierNetwork(self.image_size)

    def test_forward_pass(self):
        """
        Tests the forward pass of the ClassifierNetwork to ensure that it produces
        the correct output shape.

        The output shape should be (batch_size, num_classes).
        """
        input_tensor = torch.randn(1, 3, *self.image_size)
        output = self.network(input_tensor)
        self.assertEqual(output.shape, (1, 2), "Output shape should be (batch_size, num_classes)")

    def test_flattened_size_computation(self):
        """
        Tests the computation of the flattened size for the input image.

        The flattened size should be a positive integer.
        """
        flattened_size = self.network._compute_flattened_size(self.image_size)
        self.assertIsInstance(flattened_size, int, "Flattened size should be an integer.")
        self.assertGreater(flattened_size, 0, "Flattened size should be greater than 0.")


class TestModel(unittest.TestCase):
    """
    Unit tests for the Model class, which integrates the neural network and training logic.

    Tests include:
        - Model initialization.
        - Training loop execution.
        - Prediction and accuracy calculation.
    """
    def setUp(self):
        """
        Sets up the test environment by initializing a ClassifierNetwork and Model instance,
        along with dummy datasets and dataloaders for training and testing.
        """
        self.image_size = (32, 32)
        self.network = ClassifierNetwork(self.image_size)
        self.epochs = 5
        self.model = Model(self.network, self.epochs)

        # Create a dummy dataset and dataloader
        self.train_dataset = DummyDataset(num_samples=10, image_size=self.image_size)
        self.test_dataset = DummyDataset(num_samples=5, image_size=self.image_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=2)

    def test_model_initialization(self):
        """
        Tests the initialization of the Model class to ensure that the model
        is correctly configured with the specified number of epochs and device.
        """
        self.assertEqual(self.model.num_epochs, self.epochs, "Number of epochs should match.")
        self.assertIsNotNone(self.model.device, "Device should be initialized.")

    def test_training(self):
        """
        Tests the training loop of the Model class by mocking the optimizer
        and scheduler to prevent actual updates. The training loop should
        run without errors.
        """
        # Mock the optimizer and scheduler to prevent actual updates
        self.model.optimizer.zero_grad = MagicMock()
        self.model.optimizer.step = MagicMock()
        self.model.scheduler.step = MagicMock()

        # Run training and ensure no exceptions occur
        self.model.train(self.train_loader)

    def test_prediction(self):
        """
        Tests the prediction loop of the Model class to ensure that the
        correct number of predictions are returned and that the labels
        are within the expected range.
        """
        # Run prediction and verify the returned results
        results = self.model.predict(self.test_loader)
        self.assertEqual(len(results), len(self.test_dataset), "Results should match the number of test samples.")
        for path, (true_label, pred_label) in results.items():
            self.assertIn(true_label, [0, 1], "True label should be 0 or 1.")
            self.assertIn(pred_label, [0, 1], "Predicted label should be 0 or 1.")

    def test_accuracy_calculation(self):
        """
        Tests the accuracy calculation within the predict method of the Model class.
        A mock test loader is used to ensure deterministic accuracy calculation.
        """
        # Mock the test loader to ensure deterministic accuracy
        self.test_loader = MagicMock()
        self.test_loader.__iter__.return_value = [
            (torch.zeros(1, 3, *self.image_size), torch.tensor([0]), ["dummy_path"])
        ]
        accuracy = self.model.predict(self.test_loader)
        self.assertIn("dummy_path", accuracy, "Result should include the test sample path.")


if __name__ == "__main__":
    unittest.main()
