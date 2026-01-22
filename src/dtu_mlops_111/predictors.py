import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from loguru import logger
import wandb
from typing import Union, Tuple, List, Optional


class CustomnnUNetPredictor(nnUNetPredictor):
    """Custom nnU-Net predictor with W&B logging and Loguru support."""
    
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = None,
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 log_file_path: Optional[str] = None):
        
        # Auto-detect device: CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
                # CPU doesn't benefit from perform_everything_on_device
                perform_everything_on_device = False
        
        # Configure loguru
        if log_file_path:
            logger.add(log_file_path, rotation="100 MB")
        
        logger.info(f"Custom predictor initialized: running on {device}")
        
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm
        )
        
        # Initialize W&B for inference tracking
        if wandb.run is None:
            wandb.init(project="semantic_segmentation_nnunet_inference", 
                      config={
                          "device": str(device),
                          "tile_step_size": tile_step_size,
                          "use_mirroring": use_mirroring
                      })
            logger.info("W&B tracking initialized for inference")
    
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                            use_folds: Union[Tuple[Union[int, str]], None],
                                            checkpoint_name: str = 'checkpoint_final.pth'):
        """Initialize predictor from trained model with logging."""
        logger.info(f"Loading model from: {model_training_output_dir}")
        logger.info(f"Using folds: {use_folds}, checkpoint: {checkpoint_name}")
        
        result = super().initialize_from_trained_model_folder(
            model_training_output_dir, use_folds, checkpoint_name
        )
        
        logger.success("Model loaded successfully")
        
        # Log model info to W&B
        if wandb.run:
            wandb.config.update({
                "model_folder": model_training_output_dir,
                "folds": str(use_folds),
                "checkpoint": checkpoint_name
            })
        
        return result
    
    def predict_from_files(self, input_folder: str, output_folder: str, 
                          save_probabilities: bool = False, overwrite: bool = True,
                          num_processes_preprocessing: int = 3,
                          num_processes_segmentation_export: int = 3,
                          folder_with_segs_from_prev_stage: Optional[str] = None,
                          num_parts: int = 1, part_id: int = 0):
        """Predict with logging and W&B tracking."""
        logger.info(f"Starting inference: {input_folder} → {output_folder}")
        logger.info(f"Preprocessing processes: {num_processes_preprocessing}, "
                   f"Export processes: {num_processes_segmentation_export}")
        
        # Log inference start to W&B
        if wandb.run:
            wandb.log({"inference_status": "started", "input_folder": input_folder})
        
        result = super().predict_from_files(
            input_folder, output_folder, save_probabilities, overwrite,
            num_processes_preprocessing, num_processes_segmentation_export,
            folder_with_segs_from_prev_stage, num_parts, part_id
        )
        
        logger.success(f"Inference completed: results saved to {output_folder}")
        
        # Log inference completion to W&B
        if wandb.run:
            wandb.log({"inference_status": "completed"})
            wandb.finish()
        
        return result
