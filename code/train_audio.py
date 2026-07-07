import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from audio_loader import MIMII_AcousticDataset
from cpu_model import AcousticAnomalyDetector

# --- Configuration ---
DATA_DIR = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
MACHINES_TO_TRAIN = ["fan", "pump", "slider", "valve"] 
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def train_and_test_machine(machine_type, results_dict):
    print(f"\n{'='*50}")
    print(f"[{time.strftime('%H:%M:%S')}] 80/20 TRAIN/TEST SPLIT FOR: {machine_type.upper()}")
    print(f"{'='*50}")
    
    device = torch.device('cpu')
    
    # 1. Load Full Dataset
    full_dataset = MIMII_AcousticDataset(data_dir=DATA_DIR, machine_type=machine_type)
    
    # 2. Split 80% Train / 20% Test securely
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Lock the random seed to guarantee reproducibility for the paper
    generator = torch.Generator().manual_seed(42) 
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    print(f"Split completed: {train_size} Training samples | {test_size} Unseen Testing samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = AcousticAnomalyDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Train strictly on the 80% split
    print(f"\nInitiating Training Phase (10 Epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * mfccs.size(0)
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {running_loss/train_size:.4f}")

    # 4. Evaluate exclusively on the 20% UNSEEN Test Data
    print(f"\nInitiating Unseen Data Evaluation...")
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * mfccs.size(0)
            
            predictions = torch.sigmoid(outputs) >= 0.5
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    final_test_acc = (correct / total) * 100
    final_test_loss = test_loss / total
    
    print(f"✅ FINAL UNSEEN TESTING ACCURACY: {final_test_acc:.2f}% | Loss: {final_test_loss:.4f}")
    results_dict[machine_type] = {'acc': final_test_acc, 'loss': final_test_loss}

    # 5. Save the properly validated testing weights
    wdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights", "validated")
    os.makedirs(wdir, exist_ok=True)
    save_path = os.path.join(wdir, f"cpu_audio_{machine_type}_validated.pth")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    results = {}
    for machine in MACHINES_TO_TRAIN:
        train_and_test_machine(machine, results)
        
    # Overwrite the master log file with the real test metrics automatically
    log_path = os.path.join(DATA_DIR, "acoustic_testing_metrics.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("🎧 FINAL UNSEEN TESTING METRICS (MIMII -6dB) - 80/20 SPLIT\n")
        f.write("============================================================\n")
        f.write("Hardware: AMD Ryzen 5 7500F (CPU-Bound 1D-CNN)\n\n")
        for m in MACHINES_TO_TRAIN:
            f.write(f"[{m.upper()}] Test Loss: {results[m]['loss']:.4f} | Test Accuracy: {results[m]['acc']:.2f}%\n")
    print(f"\n[💾] True unseen metrics saved securely to {log_path}. Pipeline completely done.")