import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Path ke metadata.csv (di folder utama AuxiliaryASR)
metadata_path = r'Data_swara/metadata_clean.csv'
# Path ke folder Data untuk output
data_folder = 'New_data'
path = r'/kaggle/working/audio/res/'
os.makedirs(data_folder, exist_ok=True)  # Pastikan folder Data ada

# Baca metadata.csv menggunakan pandas
df = pd.read_csv(metadata_path)

# Ambil kolom yang diperlukan: filename, text, encoded_speaker
data = df[['filename', 'text', 'encoded_speaker']]

# Proporsi: 80% train, 10% val, 10% test
# Pertama, split menjadi train (80%) dan temp (20%)
train_data, temp_data = train_test_split(
    data, 
    test_size=0.2,  # 20% untuk val + test
    random_state=42
)

# Kedua, split temp menjadi val (50% dari temp, yaitu 10% total) dan test (50% dari temp, yaitu 10% total)
val_data, test_data = train_test_split(
    temp_data, 
    test_size=0.5,  # 50% dari temp untuk test
    random_state=42
)

# Tulis train_list.txt
with open(os.path.join(data_folder, 'train_list.txt'), 'w', encoding='utf-8') as f:
    for _, row in train_data.iterrows():
        filename = row['filename']
        text = row['text']
        speaker = row['encoded_speaker']
        f.write(f"{os.path.join(path, filename)}|{text}|{speaker}\n")

# Tulis val_list.txt
with open(os.path.join(data_folder, 'val_list.txt'), 'w', encoding='utf-8') as f:
    for _, row in val_data.iterrows():
        filename = row['filename']
        text = row['text']
        speaker = row['encoded_speaker']
        f.write(f"{os.path.join(path, filename)}|{text}|{speaker}\n")

# Tulis test_list.txt
with open(os.path.join(data_folder, 'test_list.txt'), 'w', encoding='utf-8') as f:
    for _, row in test_data.iterrows():
        filename = row['filename']
        text = row['text']
        speaker = row['encoded_speaker']
        f.write(f"{os.path.join(path, filename)}|{text}|{speaker}\n")

print(f"Train list created with {len(train_data)} entries.")
print(f"Val list created with {len(val_data)} entries.")
print(f"Test list created with {len(test_data)} entries.")