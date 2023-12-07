import warnings
import numpy as np
import torch
import sys 
import os 
import logging



from hw_as.trainer import Trainer
from hw_as.utils import prepare_device
from hw_as.utils.object_loading import get_dataloaders

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

@hydra.main(version_base=None, config_path="", config_name="config_hydra")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    print(f'{OmegaConf.to_yaml(cfg)}')

    logger = logging.getLogger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(cfg)
    print("DATALOADERS SETUP COMPLETED...")
    # build model architecture, then print to console
    model = instantiate(cfg["arch"])
    logger.info(model)
    print("MODEL SETUP COMPLETED...")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss = instantiate(cfg["Loss"]).to(device)
    print("LOSS SETUP COMPLETED...")
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    params = model.parameters()
    optimizer = instantiate(cfg["optimizer"], params=params)
    
    lrsched = cfg.get("lr_scheduler", None)
    if lrsched is not None:
        lr_scheduler = instantiate(cfg["lr_scheduler"], optimizer=optimizer)
    else: 
        lr_scheduler = None
    print("OPT AND LR SETUP COMPLETED...")

    trainer = Trainer(
        model=model,
        criterion=loss,
        optimizer=optimizer,
        config=cfg,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=cfg["trainer"].get("len_epoch", None)
    )
    trainer.train()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    print("start training")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()