"""
Shahadh, Raya Sattar and Al-Khateeb, Belal (2025) "Double Dual Convolutional Neural Network (D2CNN): A Deep Learning Model Based on Feature Extraction for Skin Cancer Classification," Iraqi Journal for Computer Science and Mathematics: Vol. 6: Iss. 1, Article 10.
DOI: https://doi.org/10.52866/2788-7421.1240
Available at: https://ijcsm.researchcommons.org/ijcsm/vol6/iss1/10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA, FactorAnalysis

from XAI.config import NUM_CLASSES, NUM_EPOCHS
from XAI.modeling.models.Base_Model import BaseModel


class FirstCNN(nn.Module):
    def __init__(self):
        super(FirstCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        return x


class SecondCNN(nn.Module):
    def __init__(self):
        super(SecondCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=7, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 96, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        return x


class ThirdCNN(nn.Module):
    def __init__(self):
        super(ThirdCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 96, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(96, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, padding=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        return x


class FourthCNN(nn.Module):
    def __init__(self):
        super(FourthCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        return x


class D2CNN(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES, pca_n_components=128, fa_n_components=128):
        super(D2CNN, self).__init__()
        self.first_cnn = FirstCNN()
        self.second_cnn = SecondCNN()
        self.third_cnn = ThirdCNN()
        self.fourth_cnn = FourthCNN()

        # Dimensionality reduction parameters
        self.pca_n_components = pca_n_components
        self.fa_n_components = fa_n_components

        # Fully connected classifier
        self.fc1 = nn.Linear(64, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, num_classes)

        # Initialize PCA and FA (will be fit during training)
        self.pca = None
        self.fa = None

    @staticmethod
    def name():
        return "D2CNN"

    def forward(self, x):
        # Extract features from all CNNs
        features1 = self.first_cnn(x)
        features2 = self.second_cnn(x)
        features3 = self.third_cnn(x)
        features4 = self.fourth_cnn(x)

        # Convert features to numpy for dimensionality reduction
        features1_np = features1.detach().cpu().numpy()
        features2_np = features2.detach().cpu().numpy()
        features3_np = features3.detach().cpu().numpy()
        features4_np = features4.detach().cpu().numpy()

        # Apply PCA to features from first and second CNNs
        if self.pca is None:
            combined_features = np.concatenate([features1_np, features2_np], axis=1)
            n_components = min(
                self.pca_n_components, combined_features.shape[0], combined_features.shape[1]
            )
            self.pca = PCA(n_components=n_components)
            self.pca.fit(combined_features)
        pca_features = self.pca.transform(np.concatenate([features1_np, features2_np], axis=1))

        # Apply FA to features from third and fourth CNNs
        if self.fa is None:
            combined_fa_features = np.concatenate([features3_np, features4_np], axis=1)
            n_components_fa = min(
                self.fa_n_components, combined_fa_features.shape[0], combined_fa_features.shape[1]
            )
            self.fa = FactorAnalysis(n_components=n_components_fa)
            self.fa.fit(combined_fa_features)

        fa_features = self.fa.transform(np.concatenate([features3_np, features4_np], axis=1))

        # Combine reduced features
        combined_features = np.concatenate([pca_features, fa_features], axis=1)

        # Remove constant columns before correlation calculation
        stddev = np.std(combined_features, axis=0)
        combined_features = combined_features[:, stddev > 0]
        
        # Remove duplicate features using correlation matrix
        corr_matrix = np.corrcoef(combined_features, rowvar=False)
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        to_drop = [
            column
            for column in range(combined_features.shape[1])
            if any(
                corr_matrix[column, row] > 0.95
                for row in range(column + 1, combined_features.shape[1])
            )
        ]

        if to_drop:
            combined_features = np.delete(combined_features, to_drop, axis=1)

        print (combined_features, combined_features.shape)
        combined_features = torch.from_numpy(combined_features).float().to(x.device)
        if combined_features.shape[1] != 64:
            # Calculate how many zeros need to be added
            current_size = combined_features.shape[1]

            if current_size > 64:
                # Truncate extra features
                combined_features = combined_features[:, :64]

            elif current_size < 64:
                padding_size = 64 - combined_features.shape[1]

                # Create a tensor of zeros with the required padding size
                padding = torch.zeros(combined_features.shape[0], padding_size).to(x.device)
                # Concatenate the padding to the end of the combined_features tensor
                combined_features = torch.cat([combined_features, padding], dim=1)

        # print(combined_features.shape)
        # Fully connected classifier
        x = F.relu(self.fc1(combined_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)

        return x
