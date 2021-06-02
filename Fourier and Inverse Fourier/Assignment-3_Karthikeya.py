# Part-(a)
import numpy as np
import math


def result_function(z1, z2):
    res = z1 * z2
    mag1 = np.sqrt(res.real ** 2 + res.imag ** 2)
    angl = math.atan(res.imag / res.real)
    real_part = res.real
    img_part = res.imag
    print(res, mag1, angl, real_part, img_part)

z1=complex('2+3j')
z2=complex('1+4j')
result_function(z1,z2)


# Part-(b)
import matplotlib.pyplot as plt
import numpy as np
# This function calculates DFT of a given signal
def dft(x):
    N=len(x)
    n=np.arange(N)
    k=n.reshape((N,1))
    e=np.exp(-2j * np.pi * k * n / N)
    X=np.dot(e,x)
    return X

# This code is used to sample the signal
srate=100
ts=1/srate
t=np.arange(0,1,ts)
x1=5*np.sin(2*np.pi*3*t)
x2=3*np.sin(2*np.pi*6*t)
x=x1+x2
X=dft(x)
N=len(X)
n=np.arange(N)
T=N/srate
freq=n/T
# Plot the signal
plt.stem(freq, abs(X), 'r')
plt.xlabel('Freq')
plt.ylabel('DFT Amplitude')
plt.show()




# part-(3)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from PIL import Image, ImageFilter
im = cv2.imread('astr.png', 0)
img=color.rgb2gray(im)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
mag_spectrum, phasee = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('input_image')
plt.subplot(2, 2, 2), plt.imshow(mag_spectrum)
plt.title('magnitude spectrum')
plt.subplot(2, 2, 3), plt.imshow(phasee)

rows,cols=img.shape
nrow,ncol=rows//2,cols//2
mask=np.zeros((rows,cols,2),np.uint8)
mask[nrow-30:nrow+30,ncol-30:ncol+30]=1
fourr_blur=dft_shift*mask
f_ishift= np.fft.ifftshift(fourr_blur)
o_img=cv2.idft(f_ishift)
mag_spectrum1, phasee = cv2.cartToPolar(o_img[:, :, 0], o_img[:, :, 1])
plt.subplot(2,2,4),plt.imshow(mag_spectrum1,cmap='gray')
plt.show()


# part-(4)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color

im = cv2.imread('astr.png', 0)
img=color.rgb2gray(im)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
mag_spectrum, phasee = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('input_image')
plt.subplot(2, 2, 2), plt.imshow(mag_spectrum)
plt.title('magnitude spectrum')
plt.subplot(2, 2, 3), plt.imshow(phasee)

rows,cols=img.shape
nrow,ncol=rows//2,cols//2
mask=np.zeros((rows,cols,2),np.uint8)
for i in range(nrow-30,nrow+30):
        for j in range(ncol-30,ncol+30):
            mask[nrow-30:nrow+30,ncol-30:ncol+30]=np.exp(-1*(i*i)+(j*j))/2*60

G_blur=dft_shift*mask
f_ishift= np.fft.ifftshift(G_blur)
o_img=cv2.idft(f_ishift)
mag_spectrum1, phasee = cv2.cartToPolar(o_img[:, :, 0], o_img[:, :, 1])
plt.subplot(2,2,4),plt.imshow(mag_spectrum1,cmap='gray')
plt.show()


# part-(5)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from PIL import Image, ImageFilter
im = cv2.imread('astr.png', 0)
img=color.rgb2gray(im)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
mag_spectrum, phasee = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('input_image')
plt.subplot(2, 2, 2), plt.imshow(mag_spectrum)
plt.title('magnitude spectrum')
plt.subplot(2, 2, 3), plt.imshow(phasee)

rows,cols=img.shape
nrow,ncol=rows//2,cols//2
dft_shift[nrow-30:nrow+30,ncol-30:ncol+30]=0

f_ishift= np.fft.ifftshift(dft_shift)
o_img=cv2.idft(f_ishift)
mag_spectrum1, phasee = cv2.cartToPolar(o_img[:, :, 0], o_img[:, :, 1])
plt.subplot(2,2,4),plt.imshow(mag_spectrum1)
plt.show()
