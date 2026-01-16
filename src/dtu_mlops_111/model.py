import os
import torch
import numpy as np
from torch import nn
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class Model(nn.Module):
    """
    Model wrapper for nnU-Net inference.
    Inherits from nn.Module for compatibility, but primarily uses nnUNetPredictor.
    """
    def __init__(self):
        super().__init__()
        # Initialize predictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=torch.cuda.is_available(),
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        
        # Path to the specific trainer/plans folder
        model_folder = os.path.join(
            "nnUNet_results", 
            "Dataset101_DroneSeg", 
            "nnUNetTrainer__nnUNetPlans__2d"
        )
        
        # Check if fold 0 checkpoint exists
        checkpoint_path = os.path.join(model_folder, "fold_0", "checkpoint_best.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}")
            self.predictor.initialize_from_trained_model_folder(
                model_folder,
                use_folds=(0,),
                checkpoint_name='checkpoint_best.pth',
            )
            self.loaded = True
        else:
            print(f"Warning: Model checkpoint not found at {checkpoint_path}. Running in dummy mode.")
            self.loaded = False
            # Dummy layer for structure compatibility
            self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If model is loaded, we might not use this directly for inference because nnUNet expects numpy and handles pre/post processing.
        This remains for compatibility or dummy usage.
        """
        if self.loaded:
            # If we wanted to use the internal network:
            # return self.predictor.network(x)
            # But usually we want the full pipeline.
            raise NotImplementedError("Use .predict(image_np) for nnU-Net inference")
        else:
            return self.layer(x)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on a single image (C, H, W) or (1, C, H, W).
        Returns prediction mask (H, W).
        """
        if not self.loaded:
             # Dummy prediction
             print("Model not loaded, returning dummy prediction")
             # Return random mask (1, H, W)
             if image.ndim == 3:
                 _, h, w = image.shape
             else:
                 _, _, h, w = image.shape
             return np.random.randint(0, 6, (1, h, w))

        # nnUNet expects (C, Z, H, W) where Z=1 for 2D plans usually
        if image.ndim == 4:
             # If (1, C, H, W), take (C, H, W) then expand
             if image.shape[0] == 1:
                 image = image[0]
        
        if image.ndim == 3:
            # (C, H, W) -> (C, 1, H, W)
            image = image[:, None, :, :]
            
        # Dummy properties
        props = {'spacing': [999, 1, 1]}
        
        # ret is (predicted_segmentation, probability_map) or just segmentation
        ret = self.predictor.predict_single_npy_array(
            image, 
            props, 
            None, 
            None, 
            False
        )
        
        # If output is (1, H, W) (pseudo-3D), squeeze to (H, W)
        if hasattr(ret, 'shape') and ret.ndim == 3 and ret.shape[0] == 1:
             ret = ret[0]
             
        return ret

    @property
    def metadata(self) -> dict:
        return {
            "name": "nnU-Net",
            "version": "2.6.2",
            "description": "Drone semantic segmentation",
            "input_shape": "(1, 3, H, W)",
            "output_shape": "(1, 1, H, W)", 
            "framework": "PyTorch",
            "license": "MIT"
        }

if __name__ == "__main__":
    model = Model()
    if model.loaded:
        print("NNUNet Model loaded.")
    else:
        x = torch.rand(1)
        print(f"Output shape of dummy model: {model(x).shape}")
