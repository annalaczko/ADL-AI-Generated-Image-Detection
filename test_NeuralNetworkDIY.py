import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from unittest.mock import MagicMock
import numpy as np
import torchvision

from NeuralNetworkDIY import ClassifierNetwork, Model

class DummyDataset(Dataset):
    """A dummy dataset for testing purposes."""
    def __init__(self, num_samples=10, image_size=(32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, *self.image_size)  # Random tensor simulating an image
        label = torch.randint(0, 2, (1,)).item()  # Random binary label
        path = f"image_{idx}.jpg"  # Dummy path
        return image, label, path


class TestClassifierNetwork(unittest.TestCase):
    def setUp(self):
        self.image_size = (32, 32)
        self.network = ClassifierNetwork(self.image_size)

    def test_forward_pass(self):
        """Test the forward pass of the network."""
        input_tensor = torch.randn(1, 3, *self.image_size)
        output = self.network(input_tensor)
        self.assertEqual(output.shape, (1, 2), "Output shape should be (batch_size, num_classes)")

    def test_flattened_size_computation(self):
        """Test the computation of the flattened size."""
        flattened_size = self.network._compute_flattened_size(self.image_size)
        self.assertIsInstance(flattened_size, int, "Flattened size should be an integer.")
        self.assertGreater(flattened_size, 0, "Flattened size should be greater than 0.")


class TestModel(unittest.TestCase):
    def setUp(self):
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
        """Test if the model is initialized correctly."""
        self.assertEqual(self.model.num_epochs, self.epochs, "Number of epochs should match.")
        self.assertIsNotNone(self.model.device, "Device should be initialized.")

    def test_training(self):
        """Test the training loop."""
        # Mock the optimizer and scheduler to prevent actual updates
        self.model.optimizer.zero_grad = MagicMock()
        self.model.optimizer.step = MagicMock()
        self.model.scheduler.step = MagicMock()

        # Run training and ensure no exceptions occur
        self.model.train(self.train_loader)

    def test_prediction(self):
        """Test the prediction loop."""
        # Run prediction and verify the returned results
        results = self.model.predict(self.test_loader)
        self.assertEqual(len(results), len(self.test_dataset), "Results should match the number of test samples.")
        for path, (true_label, pred_label) in results.items():
            self.assertIn(true_label, [0, 1], "True label should be 0 or 1.")
            self.assertIn(pred_label, [0, 1], "Predicted label should be 0 or 1.")

    def test_accuracy_calculation(self):
        """Test accuracy calculation in the predict method."""
        # Mock the test loader to ensure deterministic accuracy
        self.test_loader = MagicMock()
        self.test_loader.__iter__.return_value = [
            (torch.zeros(1, 3, *self.image_size), torch.tensor([0]), ["dummy_path"])
        ]
        accuracy = self.model.predict(self.test_loader)
        self.assertIn("dummy_path", accuracy, "Result should include the test sample path.")


if __name__ == "__main__":
    unittest.main()
