from PIL import Image
import numpy as np

try:
    img = Image.open("frontend/assets/earth_highres.png")
    print(f"Image loaded. Size: {img.size}. Mode: {img.mode}")
    img_gray = img.convert('L')
    arr = np.array(img_gray)
    print(f"Array shape: {arr.shape}")
    print(f"Min/Max: {arr.min()}/{arr.max()}")
except Exception as e:
    print(f"Failed: {e}")
