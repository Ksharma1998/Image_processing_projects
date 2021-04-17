"""The Following Program implements image registration using two different functions
one to rotate and evaluate  angular rotation and other for tx,ty(translation)
It is important to note that we have initially aligned both images to one center of mass and completed optimal rotation
and then translated it to the fixed image  location all the techniques here use brute force"""

# Importing all the required packages
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from heapq import (heappop, heappush, heapify)
import cv2 as cv
import seaborn as sns

# Reading the images
img_ref = imread('Image_20449.tif')
img_mov = imread('Image_20450.tif')


# Function to calculate SSD value for a given angle
def ssd_calc_angle(theta):
    if img_ref.shape != img_mov.shape:
        print("Images don't have the same shape.")
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
# Brute Force approach involving using all the angles
height, width = img_mov.shape
ssdq = []
heapify(ssdq)
ssdqt = []
heapify(ssdqt)
for theta in np.arange(0, 360):
    ssd1 = ssd_calc_angle(theta)
    print(theta)
    print(ssd1)
    heappush(ssdq, (ssd1, theta))

# Plotting with minimum angle
min_ssd1, min_angle = heappop(ssdq)
print(min_angle)
print(min_ssd1)
final_img = ndimage.rotate(img_mov_1, min_angle, reshape=False, mode='nearest')
final1 = img_ref + final_img
plt.imshow(final1)
plt.show()
final2 = img_ref_1 + final_img
plt.imshow(final2)
plt.show()

# Plotting the values on a graph
ssd_l = []
angle_l = []
while ssdq:
    ssd1, angle = heappop(ssdq)
    ssd_l.append(ssd1)
    angle_l.append(angle)
plt.plot(angle_l, ssd_l)
plt.xlabel('angle')
plt.ylabel('ssd')
plt.show()

# center of mass of rotated image
rot_ind = np.nonzero(final_img)
center_img_rot = [round(rot_ind[0].mean()), round(rot_ind[1].mean())]
print(center_img_rot)


# Translation to fixed image part

def ssd_calc_trans(tx, ty):
    return np.sum((np.array(img_ref, dtype=np.float32) - np.array(
        ndimage.shift(final_img, (tx - center_img_rot[0], ty - center_img_rot[1]), mode='nearest'),
        dtype=np.float32)) ** 2)


# Loop is running for longer time hence we only loop through few values for clear graph increase the range of loop
for tx in np.arange(center_img_ref[0] - 3, center_img_ref[0] + 3):
    for ty in np.arange(center_img_ref[1] - 3, center_img_ref[1] + 3):
        ssd2 = ssd_calc_trans(tx, ty)
        print(tx, ty)
        print(ssd2)
        heappush(ssdqt, (ssd2, tx, ty))

'''final_img_shifting=ndimage.shift(final_img,((center_img_ref[0]-center_img_rot[0]),center_img_ref[1]-center_img_rot[1]),mode='nearest')
plt.imshow(final_img_shifting)
plt.show()
final_shiiift=img_ref+final_img_shifting
plt.imshow(final_shiiift)
plt.show()'''

# Plotting with minimum translation value

min_ssd_t, min_tx, min_ty = heappop(ssdqt)
print(min_tx, min_ty)
print(min_ssd_t)
final_img_3 = ndimage.shift(final_img, (min_tx - center_img_rot[0], min_ty - center_img_rot[1]), mode='nearest')
plt.imshow(final_img_3)
plt.show()
final2 = img_ref + final_img_3
plt.imshow(final2)
plt.show()

ssd2l = []
tx_l = []
ty_l = []
while ssdqt:
    ssd2, tx, ty = heappop(ssdqt)
    ssd2l.append(ssd2)
    tx_l.append(tx)
    ty_l.append(ty)
plt.plot(tx_l, ssd2l)
plt.xlabel('tx')
plt.ylabel('ssd')
plt.show()
plt.plot(ty_l, ssd2l)
plt.xlabel('ty')
plt.ylabel('ssd')
plt.show()
