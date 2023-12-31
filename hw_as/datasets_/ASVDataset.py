import time
from tqdm.auto import tqdm
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import os
import torchaudio
import torch.nn.functional as F
import torch

class ASVDataset(Dataset):
    def __init__(self, 
                 part, 
                 cm_protocls_path, 
                 data_path, 
                 limit=None):
        self.part = part
        self.cm_protocls_path = cm_protocls_path
        self.data_path = data_path
        self.limit = limit
        
        self.path_to_audio = self.data_path + "_" + self.part + "/flac" 
        self.data_audio = os.listdir(self.path_to_audio)
        
        parts = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt", 
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        }

        self.path_to_protocols = self.cm_protocls_path + "/" + parts[part]

        with open(self.path_to_protocols, "r") as f:
            self.protocols = f.readlines()
        
        self.audio2target = list(map(lambda x: (x.split(" ")[1] + ".flac", x.split(" ")[-1][:-1]), self.protocols))

        if self.limit is not None:
            self.audio2target = self.audio2target[:limit]

    def __len__(self):
        return len(self.audio2target)

    def __getitem__(self, idx):
        max_wav_len = 64000
        audio = torchaudio.load(self.path_to_audio + "/" + self.audio2target[idx][0])[0].reshape(-1)
        if len(audio) > max_wav_len:
            audio = audio[:max_wav_len]
        else:
            pad_value = max_wav_len - len(audio)
            factor = pad_value // len(audio)
            cat_audio = torch.cat([audio]*(factor + 2), dim=-1)
            audio = cat_audio[:max_wav_len]
        return {
            "target": 1 if self.audio2target[idx][-1] == "bonafide" else 0,
            "audio": audio
        }