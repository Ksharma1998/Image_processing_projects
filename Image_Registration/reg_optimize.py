"""The Following Program implements image registration using two different functions one to rotate and evaluate we
use Golden search optimizer for angular rotation and other for tx,ty we use Powell optimizer for the second part It
is important to note that we have initially aligned both images to one center of mass and completed optimal rotation
and then translated it to the fixed image  location """

# Importing all the required packages
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from heapq import (heappop, heappush, heapify)
import cv2 as cv
import seaborn as sns
from scipy import optimize

# Reading the images
img_ref = imread('Image_20449.tif')
img_mov = imread('Image_20450.tif')

# Function to calculate SSD value for a given angle
ssd_al = []
a_l = []


def ssd_calc_angle(theta):
    if img_ref.shape != img_mov.shape:
        print("Images don't have the same shape.")
    ssd_al.append(np.sum((np.array(img_ref_1, dtype=np.float32) - np.array(
        ndimage.rotate(img_mov_1, theta, reshape=False, mode='nearest'), dtype=np.float32)) ** 2))
    a_l.append(theta)
    return np.sum((np.array(img_ref_1, dtype=np.float32) - np.array(
        ndimage.rotate(img_mov_1, theta, reshape=False, mode='nearest'), dtype=np.float32)) ** 2)


# center of mass and thresholding of the  of fixed image
img_ref[img_ref < 350] = 0
img_ref_ind = np.nonzero(img_ref)
center_img_ref = [round(img_ref_ind[0].mean()), round(img_ref_ind[1].mean())]
img_ref_1 = ndimage.shift(img_ref,
                          ((len(img_ref) // 2 - center_img_ref[0]), (len(img_ref[0]) // 2 - center_img_ref[1])),
                          mode='nearest')
plt.imshow(img_ref + img_ref_1)
plt.show()
print(center_img_ref)
plt.imshow(img_ref + img_mov)
plt.show()

# Center Of mass shifting and thresholding of the moving image
img_mov[img_mov < 350] = 0
img_mov_ind = np.nonzero(img_mov)
center_img_mov = [round(img_mov_ind[0].mean()), round(img_mov_ind[1].mean())]
print(center_img_mov)

img_mov_1 = ndimage.shift(img_mov,
                          ((len(img_ref) // 2 - center_img_mov[0]), (len(img_ref[0]) // 2 - center_img_mov[1])),
                          mode='nearest')
plt.imshow(img_mov + img_mov_1)
plt.show()
plt.imshow(img_mov_1 + img_mov + img_ref + img_ref_1)
plt.show()
# Verifying the Translations
plt.imshow(img_mov + img_ref)
plt.show()

result = optimize.minimize_scalar(ssd_calc_angle, method='Golden')
# Plotting with minimum angle
min_angle = result['x']
print(min_angle)
final_img = ndimage.rotate(img_mov_1, min_angle, reshape=False, mode='nearest')
final1 = img_ref + final_img
plt.imshow(final1)
plt.show()
final2 = img_ref_1 + final_img
plt.imshow(final2)
plt.show()
'''
Plotting the values on a graph
ssd_l = []
angle_l = []

plt.plot(a_l,ssd_l)
plt.xlabel('angle')
plt.ylabel('ssd')
plt.show()
'''
# center of mass of rotated image
rot_ind = np.nonzero(final_img)
center_img_rot = [round(rot_ind[0].mean()), round(rot_ind[1].mean())]
print(center_img_rot)


# Translation to fixed image part

def ssd_calc_trans(txty):
    tx, ty = txty
    return np.sum((np.array(img_ref, dtype=np.float32) - np.array(
        ndimage.shift(final_img, (tx - center_img_rot[0], ty - center_img_rot[1]), mode='nearest'),
        dtype=np.float32)) ** 2)


result = optimize.minimize(ssd_calc_trans, (center_img_rot), method="Powell")
print(result)
min_tx, min_ty = result.x

'''final_img_shifting=ndimage.shift(final_img,((center_img_ref[0]-center_img_rot[0]),center_img_ref[1]-center_img_rot[1]),mode='nearest')
plt.imshow(final_img_shifting)
plt.show()
final_shiiift=img_ref+final_img_shifting
plt.imshow(final_shiiift)
plt.show()'''

# Plotting with minimum translation value

print(min_tx, min_ty)
final_img_3 = ndimage.shift(final_img, (min_tx - center_img_rot[0], min_ty - center_img_rot[1]), mode='nearest')
plt.imshow(final_img_3)
plt.show()
final2 = img_ref + final_img_3
plt.imshow(final2)
plt.show()
'''
ssd2l = []
tx_l = []
ty_l = []
while ssdqt:
    ssd2, tx, ty = heappop(ssdqt)
    ssd2l.append(ssd2)
    tx_l.append(tx)
    ty_l.append(ty)
plt.plot(tx_l,ssd2l)
plt.xlabel('tx')
plt.ylabel('ssd')
plt.show()
plt.plot(ty_l, ssd2l)
plt.xlabel('ty')
plt.ylabel('ssd')
plt.show()
'''
