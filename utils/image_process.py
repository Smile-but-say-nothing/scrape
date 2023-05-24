from PIL import Image, ImageOps
from skimage import morphology
import numpy as np

def remove_small_object(image, image_rgb):
    # 传入灰度图像PIL
    image_np = np.array(image)
    image_np[image_np == 0] = False
    image_np[image_np != 0] = True
    image_np = image_np.astype(bool)
    mask = morphology.remove_small_objects(image_np, 5e4, connectivity=2)
    # image_np是个true和false的mask图
    temp = Image.new('RGB', image_rgb.size, (0, 0, 0))
    image_res = Image.composite(image_rgb, temp, Image.fromarray(mask))
    # image_res.show()
    return image_res

def seg(image, padding=(0, 0, 0, 0)):
    # 传入np数组
    bbox = image.getbbox()
    left = bbox[0] - padding[0]
    top = bbox[1] - padding[1]
    right = bbox[2] + padding[2]
    bottom = bbox[3] + padding[3]
    cropped_image = image.crop((left, top, right, bottom))
    # 800， 1000
    image = cropped_image.resize((900, 1200))
    return image