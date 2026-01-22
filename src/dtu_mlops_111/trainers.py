import torch
import wandb
from loguru import logger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_5epochs_custom(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = None):
        
        # Auto-detect device: CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 1. Configure loguru to write to file
        logger.add(f"{self.output_folder}/training_loguru_{{time}}.log", rotation="100 MB")
        
        # 2. Log system events
        logger.info(f"Custom trainer started: running on {device} with a target of 5 epochs.")
        
        # 3. Initialize W&B
        if wandb.run is None:
            wandb.init(project="semantic_segmentation_nnunet", config=plans)
        
        # Reduce the number of epochs from the library default (1000) to 5
        self.num_epochs = 5

    def on_epoch_end(self):
        # Send metrics to W&B and Loguru at the end of each epoch
        current_epoch = self.current_epoch
        loss = self.logger.my_fantastic_logging['train_losses'][-1]
        
        # Write to terminal/file
        logger.info(f"Epoch {current_epoch} finished. Loss: {loss:.4f}")
        
        # Send to cloud
        wandb.log({"epoch": current_epoch, "train_loss": loss})
        
        super().on_epoch_end()

    def on_train_end(self):
        logger.success("Training completed successfully!")
        wandb.finish()
        super().on_train_end()