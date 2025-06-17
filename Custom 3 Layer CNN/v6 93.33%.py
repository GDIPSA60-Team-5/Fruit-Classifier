import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CNN 3-layer Model done By Paul ---

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)


# --- Configuration ---
CONFIG = {
    "train_dir": "dataset/train_cleaned",
    "test_dir": "dataset/test",
    "batch_size": 64,
    "image_size": 224,
    "num_epochs": 30,
    "learning_rate": 0.001,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "class_names": [
        "apple",
        "banana",
        "orange",
        "mixed",
    ],
}

CONFIG["num_classes"] = len(CONFIG["class_names"])


# --- Dataset and Transforms ---
class FruitDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        indices: List[int] = None,
        transform=None,
        custom_transforms=None,
    ):
        self.root_dir = root_dir
        self.image_files = [
            f
            for f in os.listdir(root_dir)
            if f.endswith((".jpg")) and os.path.isfile(os.path.join(root_dir, f))
        ]
        if indices is not None:
            self.image_files = [self.image_files[i] for i in indices]
        self.transform = transform
        self.custom_transforms = custom_transforms or {}

        self.label_map = {name: i for i, name in enumerate(CONFIG["class_names"])}
        self.num_classes = CONFIG["num_classes"]

        self.labels = []
        for img_name in self.image_files:
            label_str = img_name.split("_")[0].lower()
            label = self.label_map.get(label_str, -1)
            if label == -1:
                raise ValueError(
                    f"Unknown label '{label_str}' extracted from filename '{img_name}'. "
                    f"Expected one of: {list(self.label_map.keys())}"
                )
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        label_str = img_name.split("_")[0].lower()
        label = self.label_map.get(label_str, -1)

        if label == -1:
            raise ValueError(
                f"Unknown label '{label_str}' extracted from filename '{img_name}'. "
                f"Expected one of: {list(self.label_map.keys())}"
            )

        # Use custom transform if available for this class
        if self.custom_transforms and label_str in self.custom_transforms:
            img = self.custom_transforms[label_str](img)
        elif self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size: int, train: bool = True):
    transform_list = []

    if train:
        transform_list += [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.8, 1.4),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomPerspective(
                distortion_scale=0.3, p=0.3, interpolation=InterpolationMode.BICUBIC
            ),
        ]
    else:
        transform_list += [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            )
        ]

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(transform_list)


def get_data_loaders(
    train_dir: str, test_dir: str, batch_size: int, image_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates and returns train, validation, and test data loaders, and class counts for the train set.
    """
    train_transform = get_transforms(image_size, train=True)
    test_transform = get_transforms(image_size, train=False)

    full_dataset = FruitDataset(train_dir, transform=train_transform)

    # Split indices for training and validation
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=full_dataset.labels,
        random_state=42,
    )

    train_dataset = FruitDataset(
        train_dir, indices=train_indices, transform=train_transform
    )
    val_dataset = FruitDataset(train_dir, indices=val_indices, transform=test_transform)
    test_dataset = FruitDataset(test_dir, transform=test_transform)

    # Calculate class counts for WeightedRandomSampler
    class_counts = {}
    for label in train_dataset.labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    num_samples = len(train_dataset)
    class_weights = [0.0] * len(class_counts)
    for cls, count in class_counts.items():
        class_weights[cls] = 1.0 / count if count > 0 else 0.0
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=6,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader, class_counts


class SimpleCNN(nn.Module):
    def __init__(self, image_size: int, num_classes: int):
        super(SimpleCNN, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        flat_size = 64 * (image_size // 8) * (image_size // 8)

        # Fully Connected Layers
        self.fc1 = nn.Linear(flat_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# --- Classifier Class ---
class FruitClassifier:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config["device"]
        self.model = SimpleCNN(config["image_size"], config["num_classes"]).to(
            self.device
        )

        # Get data loaders and class counts
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_class_counts,
        ) = get_data_loaders(
            config["train_dir"],
            config["test_dir"],
            config["batch_size"],
            config["image_size"],
        )

        # Calculate class weights (automated)
        self.class_weights = self._calculate_class_weights(
            self.train_class_counts, config["num_classes"]
        ).to(self.device)

        # # To use manual weights instead, uncomment the following line:
        # self.class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0], dtype=torch.float32).to(
        #     self.device
        # )
        # Use Focal Loss with class weights
        self.criterion = self.focal_loss

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["learning_rate"]
        )
        self.train_accuracies: List[float] = []

    def _calculate_class_weights(
        self, class_counts: Dict[int, int], num_classes: int
    ) -> torch.Tensor:
        """
        Calculates inverse frequency class weights.
        Weights are inversely proportional to class frequencies.
        Higher weight for minority classes.
        """
        total_samples = sum(class_counts.values())
        weights = torch.zeros(num_classes)

        for i, class_name in enumerate(self.config["class_names"]):
            count = class_counts.get(i, 0)
            if count > 0:
                weights[i] = total_samples / (num_classes * count)
            else:
                weights[i] = 1.0

        print("\nClass Distribution in Training Data:")
        for i, class_name in enumerate(self.config["class_names"]):
            train_count = sum(
                1 for label in self.train_loader.dataset.labels if label == i
            )
            val_count = sum(1 for label in self.val_loader.dataset.labels if label == i)
            print(
                f"- {class_name}: {train_count} samples (Training), {val_count} samples (Validation)"
            )

        print(f"\nCalculated Class Weights: {weights.tolist()}")
        return weights

    def focal_loss(self, logits, targets, gamma=2.0, reduction="mean", alpha=None):
        """
        Multiclass Focal Loss without class weights (alpha=None) or with manual alpha override.
        logits: [batch, num_classes], targets: [batch]
        """

        ce_loss = (
            F.cross_entropy(logits, targets, weight=alpha, reduction="none")
            if alpha is not None
            else F.cross_entropy(logits, targets, reduction="none")
        )
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** gamma) * ce_loss
        if reduction == "mean":
            return focal.mean()
        elif reduction == "sum":
            return focal.sum()
        else:
            return focal

    def _evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluates the model on a given dataset (validation or test).
        Returns the accuracy percentage.
        """
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total if total > 0 else 0

    def train(self):
        """
        Executes the full training process for the specified number of epochs.
        Saves the best model (by validation accuracy) and the final model.
        """
        best_val_acc = 0.0  # Track the best validation accuracy
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            running_loss = 0.0
            correct_preds = 0
            total_samples = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                colour="blue",
                ncols=60,
            )
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(
                    outputs, labels, gamma=2.0, alpha=self.class_weights
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)

                # Update tqdm bar with current batch loss
                progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            avg_loss = running_loss / total_samples
            train_accuracy = correct_preds / total_samples

            self.train_accuracies.append(train_accuracy * 100)

            # Evaluate on validation set
            val_accuracy = self._evaluate(self.val_loader)
            print(
                f"Epoch {epoch + 1:02d}/{self.config['num_epochs']:02d} | "
                f"Loss: {avg_loss:.5f} | "
                f"Train Acc: {train_accuracy * 100:6.2f}% | "
                f"Val Acc: {val_accuracy:6.2f}%"
            )

            # Save the best model based on validation accuracy if > 90%
            if val_accuracy > best_val_acc and val_accuracy >= 90.0:
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), "best_model.pth")
                print(
                    f"Highest validation accuracy found: {best_val_acc:.2f}% - Model saved to 'best_model.pth'"
                )

        torch.save(self.model.state_dict(), "final_model.pth")
        print("Final model saved to 'final_model.pth'")

    def evaluate(self) -> float:
        """
        Evaluates the model on the test dataset.
        Returns the accuracy percentage.
        """
        test_accuracy = self._evaluate(self.test_loader)
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")
        return test_accuracy

    def plot_training_metrics(self):
        """
        Plots the training accuracy over epochs.
        """
        epochs = range(1, self.config["num_epochs"] + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_accuracies, marker="o", label="Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix_and_report(self):
        """
        Generates and plots the confusion matrix and prints the classification report
        for the test set.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.config["class_names"],
            yticklabels=self.config["class_names"],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        print("\nClassification Report:\n")
        print(
            classification_report(
                all_labels, all_preds, target_names=self.config["class_names"]
            )
        )


def main():
    # Instantiate the classifier with the configuration
    classifier = FruitClassifier(CONFIG)

    # Train the model
    classifier.train()

    # Evaluate the model
    classifier.evaluate()

    # Plot metrics
    classifier.plot_training_metrics()
    classifier.plot_confusion_matrix_and_report()


# --- Main Execution ---
if __name__ == "__main__":
    main()
