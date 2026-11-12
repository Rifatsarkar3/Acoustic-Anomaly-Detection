import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class MIMII_AcousticDataset(Dataset):
    def __init__(self, data_dir, machine_type="fan", n_mfcc=40, max_pad_len=400):
        """
        Loads MIMII .wav files and converts them to MFCC arrays.
        """
        # --- THE FIX: Point directly to the machine_type folder ---
        self.data_dir = os.path.join(data_dir, machine_type)

        self.n_mfcc = n_mfcc
        self.max_pad_len = max_pad_len
        self.samples = []

        print(f"Scanning {machine_type} directories in {self.data_dir}...")

        # MIMII structure: machine_type / id_XX / normal (or abnormal) / .wav
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}. Did you run the extraction script?")

        for machine_id in os.listdir(self.data_dir):
            id_path = os.path.join(self.data_dir, machine_id)
            if not os.path.isdir(id_path): continue

            # Label 0.0 for Normal, 1.0 for Abnormal
            for condition, label in [("normal", 0.0), ("abnormal", 1.0)]:
                condition_path = os.path.join(id_path, condition)
                if not os.path.exists(condition_path): continue

                for f in os.listdir(condition_path):
                    if f.endswith('.wav'):
                        self.samples.append({
                            'path': os.path.join(condition_path, f),
                            'label': label,
                            'machine_id': machine_id,
                        })

        print(f"Loaded {len(self.samples)} {machine_type} audio samples.")

    def __len__(self):
        return len(self.samples)

    def _extract_mfcc(self, file_path):
        # Load audio at 16kHz
        audio, sample_rate = librosa.load(file_path, sr=16000)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)

        # Pad or truncate the sequence to ensure a uniform tensor size
        if mfcc.shape[1] > self.max_pad_len:
            mfcc = mfcc[:, :self.max_pad_len]
        else:
            pad_width = self.max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Output shape: (1, n_mfcc, max_pad_len) -> e.g., (1, 40, 400)
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mfcc_tensor = self._extract_mfcc(sample['path'])
        label = torch.tensor([sample['label']], dtype=torch.float32)
        return mfcc_tensor, label
