from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from heapq import (heappop, heappush, heapify)
from PIL import Image
import matplotlib.image as mpimg
from skimage import color
from skimage import io

img_ref = imread('Image_20449.tif')

img_mov = imread('Image_20450.tif')

# center of mass  of both images
img_ref[img_ref < 262] = 0
img_ref_ind = np.nonzero(img_ref)
center_img_ref = [round(img_ref_ind[0].mean()), round(img_ref_ind[1].mean())]
img_ref = ndimage.shift(img_ref, ((len(img_ref) // 2 - center_img_ref[0]), (len(img_ref[0]) // 2 - center_img_ref[1])),
                        mode='nearest')
plt.imshow(img_ref)
plt.show()

img_mov[img_mov < 262] = 0

img_mov_ind = np.nonzero(img_mov)
center_img_mov = [round(img_mov_ind[0].mean()), round(img_mov_ind[1].mean())]

img_mov = ndimage.shift(img_mov, ((len(img_ref) // 2 - center_img_mov[0]), (len(img_mov[0]) // 2 - center_img_mov[1])),
                        mode='nearest')

plt.imshow(img_mov + img_ref)
plt.show()

plt.imshow(img_mov)
plt.show()
height, width = img_mov.shape

ssdq = []
heapify(ssdq)

for theta in np.arange(0, 360):
    rotated = ndimage.rotate(img_mov, theta, reshape=False, mode='nearest')
    if img_ref.shape != rotated.shape:
        print("Images don't have the same shape.")
    ssd = np.sum((np.array(img_ref, dtype=np.float32) - np.array(rotated, dtype=np.float32)) ** 2)
    print(theta)
    print(ssd)
    heappush(ssdq, (ssd, theta))

ssd, angle = heappop(ssdq)
print(angle)
print(ssd)
final_img = ndimage.rotate(img_mov, angle, reshape=False, mode='nearest')

final = img_ref + final_img
plt.imshow(final)
plt.show()
ssd_l = []
angle_l = []
while ssdq:
    ssd, angle = heappop(ssdq)
    ssd_l.append(ssd)
    angle_l.append(angle)

plt.scatter(angle_l, ssd_l, label="stars", color="red", marker="*", s=30)
plt.xlabel('angle')
plt.ylabel('ssd')
plt.show()

