import torch
import torch.nn as nn

class AcousticAnomalyDetector(nn.Module):
    def __init__(self):
        super(AcousticAnomalyDetector, self).__init__()

        # Input shape from DataLoader: (Batch, 1 Channel, 40 MFCC Bins, 400 Time Frames)
        # We use Conv2D because MFCCs act like 2D spectrogram "images" of sound
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Shape becomes: (Batch, 16, 20, 200)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Shape becomes: (Batch, 32, 10, 100)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Shape becomes: (Batch, 64, 5, 50)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 50, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Prevent overfitting on the acoustic noise
            nn.Linear(128, 1) # Binary Output: 0 (Normal) vs 1 (Abnormal)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
