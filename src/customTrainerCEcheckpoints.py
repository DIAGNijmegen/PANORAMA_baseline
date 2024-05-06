import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCELoss import nnUNetTrainerCELoss

class nnUNetTrainer_Loss_CE_checkpoints(nnUNetTrainerCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.save_latest_only = False
