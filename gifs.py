import glob
from PIL import Image
import numpy as np


FPATH = "generated_im/*.png"
ENDING_FRAMES = 50
FRAME_TIME = 110
START_FRAME = 1     # 0 lub 1

IMAGES_PATHS = glob.glob(FPATH)
names = np.empty(len(IMAGES_PATHS) + ENDING_FRAMES, dtype=object)
for image_path in IMAGES_PATHS:
    names[int(str(image_path).replace("generated_im\\", "").replace(".png", "")) - START_FRAME] = image_path
names[len(IMAGES_PATHS):len(IMAGES_PATHS) + ENDING_FRAMES] = names[len(IMAGES_PATHS) - 1]
images = (Image.open(file) for file in names)
image = next(images)
image.save(fp=f"generated_im/gif.gif", format='GIF', append_images=images, save_all=True, duration=FRAME_TIME, loop=0)
