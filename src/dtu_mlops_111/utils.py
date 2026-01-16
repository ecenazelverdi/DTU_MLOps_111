import base64
import io

import numpy as np
from PIL import Image

# Class color mapping (R, G, B)
CLASS_COLORS = {
    0: (0, 0, 0),        # background
    1: (155, 38, 182),   # obstacles
    2: (14, 135, 204),   # water
    3: (124, 252, 0),    # soft-surfaces
    4: (255, 20, 147),   # moving-objects
    5: (169, 169, 169),  # landing-zones
}

def array_to_base64(array: np.ndarray) -> str:
    """
    Convert a 2D numpy array (segmentation mask) to a Base64 encoded PNG string.
    Applies color mapping based on CLASS_COLORS.
    
    Args:
        array: 2D numpy array (H, W) containing class integers.
        
    Returns:
        str: Base64 encoded string of the image.
    """
    # Create RGB image container
    h, w = array.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colors
    for class_id, color in CLASS_COLORS.items():
        mask = array == class_id
        rgb_image[mask] = color
        
    # Convert to PIL Image
    image = Image.fromarray(rgb_image)
    
    # Buffer to save image
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    # Encode
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str
