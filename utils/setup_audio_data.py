import os
import zipfile

# 1. Point to the root folder where your zip files are currently sitting
SOURCE_DIR = r"D:\All Data sets"

# 2. Extract them into our current paper's workspace
TARGET_DIR = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"

zip_files = [
    "-6_dB_fan.zip",
    "-6_dB_pump.zip",
    "-6_dB_slider.zip",
    "-6_dB_valve.zip"
]

def extract_all():
    print("Initiating MIMII Dataset Extraction...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    for zip_name in zip_files:
        zip_path = os.path.join(SOURCE_DIR, zip_name)
        
        if os.path.exists(zip_path):
            print(f"Extracting {zip_name} -> MIMII_Dataset/")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TARGET_DIR)
        else:
            print(f"[WARNING] Could not find {zip_name} in {SOURCE_DIR}")
            
    print("\nExtraction Complete! Data is staged and ready for MFCC processing.")

if __name__ == "__main__":
    extract_all()