#!/usr/bin/env python3
"""Run nnU-Net inference using custom predictor with W&B and Loguru logging."""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from dtu_mlops_111.predictors import CustomnnUNetPredictor


def main():
    parser = argparse.ArgumentParser(description="Run nnU-Net inference with custom predictor")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model directory")
    parser.add_argument("-f", "--fold", type=str, default="0", help="Fold to use")
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoint_best.pth", 
                       help="Checkpoint name")
    parser.add_argument("--disable-tta", action="store_true", 
                       help="Disable test time augmentation")
    parser.add_argument("--log-file", type=str, default=None, 
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file:
        logger.add(args.log_file, rotation="100 MB")
    
    logger.info("=" * 50)
    logger.info("Starting Custom nnU-Net Inference")
    logger.info("=" * 50)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Fold: {args.fold}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"TTA: {'Disabled' if args.disable_tta else 'Enabled'}")
    
    try:
        # Initialize custom predictor
        predictor = CustomnnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=not args.disable_tta,
            perform_everything_on_device=True,
            device=None,  # Auto-detect
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
            log_file_path=args.log_file
        )
        
        # Load model
        predictor.initialize_from_trained_model_folder(
            args.model,
            use_folds=(args.fold,),
            checkpoint_name=args.checkpoint
        )
        
        # Run inference
        predictor.predict_from_files(
            args.input,
            args.output,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=3,
            num_processes_segmentation_export=3
        )
        
        logger.success("Inference completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
