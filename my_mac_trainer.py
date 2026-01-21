import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_5epochs_Mac(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = None):
        
        # Device selection: use MPS on Mac if available, otherwise CUDA, else CPU
        if device is None or device.type == 'cuda':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        print(f"--- [MLOps Mode] Training on device: {device} ---")
        
        # Pass the selected device to the original nnU-Net constructor
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Set number of epochs to 5 (matching the library configuration)
        self.num_epochs = 5