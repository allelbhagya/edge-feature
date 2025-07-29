import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from feature import prewitt_operator, sobel_operator, scharr_operator

OUTPUT_DIR = "output"
OUTPUT_FILES = {
    "combined":"combined.png",
    "prewitt": "prewitt.png",
    "sobel": "sobel.png",
    "scharr": "scharr.png",
}

LABELS = ["Prewitt", "Sobel", "Scharr"]

def normalize_image(img):
    return (img / np.max(img) * 255).astype(np.uint8)

def main(image_path):
    image = Image.open(image_path).convert("L")
    image_np = np.array(image, dtype=np.float32)

    prewitt_edges = normalize_image(prewitt_operator(image_np))
    sobel_edges = normalize_image(sobel_operator(image_np))
    scharr_edges = normalize_image(scharr_operator(image_np))

    prewitt_img = Image.fromarray(prewitt_edges)
    sobel_img = Image.fromarray(sobel_edges)
    scharr_img = Image.fromarray(scharr_edges)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prewitt_img.save(os.path.join(OUTPUT_DIR, OUTPUT_FILES["prewitt"]))
    sobel_img.save(os.path.join(OUTPUT_DIR, OUTPUT_FILES["sobel"]))
    scharr_img.save(os.path.join(OUTPUT_DIR, OUTPUT_FILES["scharr"]))

    w, h = prewitt_img.size
    combined = Image.new('L', (w * 3, h))
    combined.paste(prewitt_img, (0, 0))
    combined.paste(sobel_img, (w, 0))
    combined.paste(scharr_img, (2 * w, 0))

    combined.save(os.path.join(OUTPUT_DIR, OUTPUT_FILES["combined"]))
    combined.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="edge detection techniques")
    parser.add_argument("--image", required=True, help="path")
    args = parser.parse_args()
    main(args.image)
