import torch
import torch.nn as nn
import torchvision.models as models

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class LightweightCNNConformer(nn.Module):
    """
    Proposed Novel Architecture: CNN + Self-Attention (Conformer-lite)
    Extracts local features with CNN, then models global context with TransformerEncoder.
    """
    def __init__(self, input_channels=1, num_classes=1, embedding_dim=128):
        super().__init__()
        # Local Feature Extraction (CNN)
        self.cnn = nn.Sequential(
            ConvolutionalBlock(input_channels, 16),
            ConvolutionalBlock(16, 32),
            ConvolutionalBlock(32, 64)
        )
        
        # Calculate shape after 3 max-pools:
        # e.g., input (1, 64, 400) -> (16, 32, 200) -> (32, 16, 100) -> (64, 8, 50)
        
        # Global Feature Modeling (Self-Attention)
        self.d_model = 64 * 8  # 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Bottleneck & Classifier
        self.fc = nn.Linear(self.d_model, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_embedding=False):
        # x: (B, 1, F, T) -> e.g. (B, 1, 64, 400)
        x = self.cnn(x) # (B, 64, F', T')
        
        B, C, F_prime, T_prime = x.shape
        # Reshape for transformer: (B, T', C * F')
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T_prime, C * F_prime)
        
        x = self.transformer(x)
        
        # Global Average Pooling over time
        x = x.mean(dim=1)
        
        emb = self.fc(self.dropout(x))
        out = self.classifier(torch.relu(emb))
        
        if return_embedding:
            return out, emb
        return out

class MobileNetV3Audio(nn.Module):
    """
    Baseline: Standard lightweight CNN (MobileNetV3 Small)
    """
    def __init__(self, num_classes=1):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=None)
        # Modify first layer for 1-channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Modify classifier
        in_features = mobilenet.classifier[0].weight.shape[1]
        mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.model = mobilenet

    def forward(self, x, return_embedding=False):
        # MobileNetV3 doesn't easily expose the embedding layer in our modification without a hook,
        # so for this baseline we only return the output.
        out = self.model(x)
        if return_embedding:
            return out, out # Placeholder embedding
        return out

class STgramMFN_Lite(nn.Module):
    """
    Baseline: Simplified STgram-MFN (Spectral-Temporal Fusion)
    Uses parallel branches for log-Mel and differences.
    """
    def __init__(self, num_classes=1, embedding_dim=128):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvolutionalBlock(1, 16),
            ConvolutionalBlock(16, 32),
            ConvolutionalBlock(32, 64)
        )
        self.branch2 = nn.Sequential(
            ConvolutionalBlock(1, 16),
            ConvolutionalBlock(16, 32),
            ConvolutionalBlock(32, 64)
        )
        # Assume input (1, 64, 400) -> output of branch is (64, 8, 50) = 25600
        self.fc1 = nn.Linear(25600 * 2, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        # branch 1: normal input
        out1 = self.branch1(x)
        
        # branch 2: temporal difference
        # pad the difference to maintain shape
        diff = torch.zeros_like(x)
        diff[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]
        out2 = self.branch2(diff)
        
        merged = torch.cat((out1.view(out1.size(0), -1), out2.view(out2.size(0), -1)), dim=1)
        emb = torch.relu(self.fc1(merged))
        out = self.classifier(emb)
        
        if return_embedding:
            return out, emb
        return out
