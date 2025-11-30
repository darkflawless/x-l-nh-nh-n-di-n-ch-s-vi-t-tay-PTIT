



base_folder = r"C:\Users\Admin\Desktop\xla\hinhhoc\shapes\circle"


import os
from PIL import Image
import numpy as np


# Load and resize circle image
circle_path = os.path.join(base_folder, "1.png")
img = Image.open(circle_path)
img_resized = img.resize((64, 64))
print(f"Resized circle 1 to 64x64: {img_resized.size}")
img_resized.show()