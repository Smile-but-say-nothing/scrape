import PIL.Image
from PIL import Image, ImageOps
import glob
from skimage import morphology
import numpy as np
import os
from io import BytesIO


# img = Image.open('./高玉洁/933 2022-09-01/背部二_seg.jpg').convert('RGB')
# img = ImageOps.exif_transpose(img)  # avoid EXIF info in images
# img = remove_small_object(img.convert('L'), img)

root = glob.glob('./*/*/*seg*.jpg')
print(root)
set_list = []
last_split_left = os.path.split(root[0])[0]
last_people_name = last_split_left.split('\\')[1]
counter = 0
for idx, path in enumerate(root):
    split_left, name = os.path.split(path)
    if split_left == last_split_left:
        set_list.append(name)
        continue
    print(last_split_left, set_list)
    set_list = [Image.open(os.path.join(last_split_left, n)) for n in set_list]
    set_list = [np.array(seg(remove_small_object(image_rgb.convert('L'), image_rgb)), dtype=np.uint8) // len(set_list) for image_rgb in set_list]
    res = np.zeros(set_list[0].shape, dtype=np.uint8)
    for M in set_list:
        res += M
    print(np.sum(res))
    res = Image.fromarray(res).convert('RGB')
    # res.show()
    # res.save('./叠加/' + last_people_name + ' ' + str(counter) + '.jpg')
    res.save('./' + last_people_name + '/' + last_people_name + ' ' + str(counter) + '.jpg')
    # 更新旧值
    people_name = split_left.split('\\')[1]
    if people_name == last_people_name:
        counter += 1
    else:
        last_people_name = people_name
        counter = 0
    set_list = [name]
    last_split_left = split_left
