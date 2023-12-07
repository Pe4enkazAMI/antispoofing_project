import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]) -> dict:
    print("IM HERE!: ", dataset_items[0]["audio"].shape, dataset_items[0]["target"])
    audios = pad_sequence([items["audio"] for items in dataset_items], batch_first=True)
    targets = torch.tensor([items["target"] for items in dataset_items])
    return {
        "audio": audios,
        "targets": targets
    }