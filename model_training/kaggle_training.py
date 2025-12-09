import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# Configuration
CONFIG = {
    'data_dir': '/kaggle/input/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'num_epochs': 50,
    'hidden_dim': 512,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 1e-3,
}

print(f"Device: {CONFIG['device']}")
print(f"PyTorch version: {torch.__version__}")



# ============================================================================
# DATA LOADING (from your previous code)
# ============================================================================

def load_split(data_dir, split='train'):
    """Load train/val/test split"""
    from glob import glob
    
    pattern = f'{data_dir}/**/data_{split}.hdf5'
    files = sorted(glob(pattern, recursive=True))
    
    print(f"\nLoading {split} split...")
    print(f"Files found: {len(files)}")
    
    all_data = {k: [] for k in ['neural', 'n_steps', 'sentence', 'phonemes', 
                                 'phoneme_len', 'session', 'block', 'trial']}
    
    for filepath in tqdm(files):
        with h5py.File(filepath, 'r') as f:
            for trial_key in f.keys():
                trial = f[trial_key]
                
                # Neural data
                neural = trial['input_features'][:]
                n_steps = trial.attrs['n_time_steps']
                
                # Metadata
                session = trial.attrs['session']
                if isinstance(session, bytes):
                    session = session.decode('utf-8')
                block = trial.attrs['block_num']
                trial_num = trial.attrs['trial_num']
                
                # Labels (train/val only)
                sentence = trial.attrs.get('sentence_label')
                if sentence and isinstance(sentence, bytes):
                    sentence = sentence.decode('utf-8')
                
                phonemes = trial.get('seq_class_ids')[:] if 'seq_class_ids' in trial else None
                phoneme_len = trial.attrs.get('seq_len')
                
                all_data['neural'].append(neural)
                all_data['n_steps'].append(n_steps)
                all_data['sentence'].append(sentence)
                all_data['phonemes'].append(phonemes)
                all_data['phoneme_len'].append(phoneme_len)
                all_data['session'].append(session)
                all_data['block'].append(block)
                all_data['trial'].append(trial_num)
    
    print(f"✓ Loaded {len(all_data['neural'])} samples")
    return all_data




# ============================================================================
# DATASET & MODEL (from Phase 1)
# ============================================================================

class BrainToTextDataset(Dataset):
    def __init__(self, data, char2idx=None, normalize=True):
        self.neural = data['neural']
        self.n_steps = data['n_steps']
        self.sentences = data['sentence']
        self.normalize = normalize
        
        if char2idx is None:
            self.char2idx = self._build_vocab()
        else:
            self.char2idx = char2idx
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
    
    def _build_vocab(self):
        chars = set()
        for sent in self.sentences:
            if sent:
                chars.update(sent.lower())
        chars = sorted(list(chars))
        char2idx = {'<BLANK>': 0}
        for i, ch in enumerate(chars, start=1):
            char2idx[ch] = i
        return char2idx
    
    def __len__(self):
        return len(self.neural)
    
    def __getitem__(self, idx):
        neural = self.neural[idx][:self.n_steps[idx]]
        
        if self.normalize:
            neural = (neural - neural.mean()) / (neural.std() + 1e-8)
        
        sentence = self.sentences[idx] if self.sentences[idx] else ""
        target = [self.char2idx.get(ch.lower(), 0) for ch in sentence]
        
        return {
            'neural': torch.FloatTensor(neural),
            'target': torch.LongTensor(target),
            'length': len(neural),
            'target_length': len(target),
            'sentence': sentence
        }




def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    neurals = [item['neural'] for item in batch]
    targets = [item['target'] for item in batch]
    
    neural_padded = pad_sequence(neurals, batch_first=True)
    target_padded = pad_sequence(targets, batch_first=True)
    lengths = torch.LongTensor([item['length'] for item in batch])
    target_lengths = torch.LongTensor([item['target_length'] for item in batch])
    
    return {
        'neural': neural_padded,
        'target': target_padded,
        'lengths': lengths,
        'target_lengths': target_lengths,
        'sentences': [item['sentence'] for item in batch]
    }
# ============================================================================  
def train_model(train_loader, val_loader, char2idx, config):
    """Train CTC model"""
    model = BaselineCTCModel(
        input_dim=512,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        vocab_size=len(char2idx),
        dropout=config['dropout']
    ).to(config['device'])
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    idx2char = {v: k for k, v in char2idx.items()}
    best_wer = float('inf')
    
    print("\nTraining...")
    for epoch in range(config['num_epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'):
            neural = batch['neural'].to(config['device'])
            target = batch['target'].to(config['device'])
            lengths = batch['lengths']
            target_lengths = batch['target_lengths']
            
            log_probs = model(neural, lengths)
            loss = criterion(log_probs, target, lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_wer = validate_model(model, val_loader, idx2char, config['device'])
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, WER={val_wer:.2f}%")
            
            if val_wer < best_wer:
                best_wer = val_wer
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'char2idx': char2idx,
                    'config': config,
                    'wer': val_wer
                }, 'best_model.pt')
                print(f"✓ Saved (WER: {val_wer:.2f}%)")
    
    return model, best_wer


