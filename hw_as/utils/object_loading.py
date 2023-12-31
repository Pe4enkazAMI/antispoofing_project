from operator import xor

from hw_as.collate_fn.collate import collate_fn
from torch.utils.data import ConcatDataset, DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig

def get_dataloaders(cfg: DictConfig):
    dataloaders = {}
    for split, params in cfg["data"].items():
        num_workers = params.get("num_workers", 1)
        print("split: ", split)
        print("params: ", params)

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            print(ds)
            datasets.append(instantiate(config=ds))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=True
        )
        dataloaders[split] = dataloader
    return dataloaders