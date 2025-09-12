import csv
import random
import os

random.seed(42)  # Tambahkan seed agar hasil shuffle konsisten

# Path ke metadata.csv (di folder utama AuxiliaryASR)
metadata_path = r'Data_swara/metadata_clean.csv'
# Path ke folder Data untuk output
data_folder = 'New_data'
path = r'/kaggle/working/audio/res/'
os.makedirs(data_folder, exist_ok=True)  # Pastikan folder Data ada

# Baca metadata.csv
data = []
with open(metadata_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        # row[3] = filename, row[2] = text, row[5] = encoded_speaker
        data.append((row[3], row[2], row[5]))

# Shuffle data untuk split acak
random.shuffle(data)

# Split 80% train, 20% val
split_index = int(0.8 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# Tulis train_list.txt
with open(os.path.join(data_folder, 'train_list.txt'), 'w', encoding='utf-8') as f:
    for filename, text, speaker in train_data:
        f.write(f"{os.path.join(path, filename)}|{text}|{speaker}\n")

# Tulis val_list.txt
with open(os.path.join(data_folder, 'val_list.txt'), 'w', encoding='utf-8') as f:
    for filename, text, speaker in val_data:
        f.write(f"{os.path.join(path, filename)}|{text}|{speaker}\n")

print(f"Train list created with {len(train_data)} entries.")
print(f"Val list created with {len(val_data)} entries.")