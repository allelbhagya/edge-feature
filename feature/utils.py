import numpy as np
from PIL import Image

def load_grayscale_image(path):
    with Image.open(path).convert('L') as img:
        return np.array(img, dtype=np.float32)
