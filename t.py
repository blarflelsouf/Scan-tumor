from PIL import Image
import numpy as np

img1 = Image.open("/home/blqrf/code/blarflelsouf/Scan-tumor/data_parent/A_raw_data/Testing/pituitary/Te-pi_0012.jpg")
img2 = Image.open("data_parent/A_raw_data/Testing/pituitary/Te-pi_0166.jpg")

print("Image 1 shape:", np.array(img1).shape)
print("Image 2 shape:", np.array(img2).shape)

if len(np.array(img2).shape) == 2:  # If image is grayscale
    img2_exp = img2.convert("RGB")

print(np.array(img2_exp).shape)
