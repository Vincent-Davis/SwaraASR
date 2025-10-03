import torch
import yaml
import os
import click
from meldataset import build_dataloader
from models import build_model
from trainer import Trainer
from utils import calc_wer
from text_utils import TextCleaner
import pandas as pd
import numpy as np

# Fungsi untuk CTC greedy decoding
def ctc_greedy_decode(logits, blank=0):
    """
    Greedy decoding untuk CTC logits.
    logits: (batch, seq_len, n_token)
    return: list of predicted sequences (list of int)
    """
    preds = torch.argmax(logits, dim=-1)  # (batch, seq_len)
    decoded = []
    for pred in preds:
        # Remove consecutive duplicates and blanks
        prev = blank
        seq = []
        for p in pred:
            if p != blank and p != prev:
                seq.append(p.item())
            prev = p
        decoded.append(seq)
    return decoded

@click.command()
@click.option('-c', '--config_path', default='./Configs/config_swara.yml', type=str)
@click.option('-d', '--data_path', required=True, type=str, help='Path to val_list.txt or test_list.txt')
@click.option('-o', '--output_path', default='inference_results.csv', type=str, help='Output file for results')
@click.option('-p', '--pretrained_model', type=str, help='Path to pretrained model checkpoint (overrides config)')
def main(config_path, data_path, output_path, pretrained_model):
    config = yaml.safe_load(open(config_path))
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_params = config.get('model_params', {})
    model = build_model(model_params)
    model.to(device)
    
    # Load checkpoint: dari parameter atau config
    checkpoint_path = pretrained_model or config.get('pretrained_model', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict['model'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint loaded, using untrained model.")
    
    model.eval()
    
    # Load vocab untuk decoding
    text_cleaner = TextCleaner()
    reverse_vocab = {v: k for k, v in text_cleaner.word_index_dictionary.items()}
    blank_index = text_cleaner.word_index_dictionary.get("$", 0)  # Pad sebagai blank
    
    # Buat dataloader: hanya kirim sr ke dataset_config
    batch_size = config.get('batch_size', 8)
    preprocess_params = config.get('preprocess_parasm', {})
    dataset_config = {'sr': preprocess_params.get('sr', 24000)}  # Hanya sr yang diterima MelDataset
    
        # Di dalam main(), sebelum build_dataloader:
    with open(data_path, 'r', encoding='utf-8') as f:
        path_list = [line.strip() for line in f.readlines() if line.strip()]  # Baca dan filter baris kosong

    dataloader = build_dataloader(path_list, batch_size=batch_size, num_workers=4, 
                              dataset_config=dataset_config, device=device)

    results = []
    total_wer = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            texts, input_lengths, mels, output_lengths = batch  # Batch structure
            mels = mels.to(device)
            
            # Forward pass: get CTC logits
            ctc_logits = model(mels)  # (batch, seq_len, n_token)
            
            # CTC decoding
            preds = ctc_greedy_decode(ctc_logits, blank=blank_index)
            
            # Decode ke text
            for i, pred_seq in enumerate(preds):
                # Predicted phoneme text
                pred_text = ''.join([reverse_vocab.get(p, '') for p in pred_seq])
                
                # Ground truth phoneme text (dari texts, tanpa blank di awal/akhir)
                gt_seq = texts[i][:input_lengths[i]].tolist()
                gt_text = ''.join([reverse_vocab.get(g, '') for g in gt_seq if g != blank_index])
                
                # Hitung WER
                wer = calc_wer(gt_text, pred_text)
                total_wer += wer
                total_samples += 1
                
                # Simpan hasil (filename tidak tersedia di batch, gunakan index atau skip)
                results.append({
                    'sample_id': total_samples - 1,
                    'ground_truth': gt_text,
                    'predicted': pred_text,
                    'wer': wer
                })
    
    # Hitung average WER
    avg_wer = total_wer / total_samples if total_samples > 0 else 0.0
    
    # Simpan hasil ke CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Inference completed. Results saved to {output_path}")
    print(f"Total samples: {total_samples}")
    print(f"Average WER: {avg_wer:.4f}")

if __name__ == "__main__":
    main()