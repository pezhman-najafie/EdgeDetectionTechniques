import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread(r'C:\Users\pezhm\Desktop\New folder\lena.PNG', cv2.IMREAD_GRAYSCALE)

# Sobel operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobel = np.sqrt(sobelx**2 + sobely**2)

# Canny edge detector
canny = cv2.Canny(img, 100, 200)

# Laplacian of Gaussian (LoG)
log = cv2.Laplacian(cv2.GaussianBlur(img, (3,3), 0), cv2.CV_64F)

# Difference of Gaussians (DoG)
gaussian1 = cv2.GaussianBlur(img, (3,3), 0)
gaussian2 = cv2.GaussianBlur(img, (9,9), 0)
dog = gaussian1 - gaussian2

# Hildreth Edge Detector (Zero-crossing of LoG)
def zero_crossing(img):
    rows, cols = img.shape
    result = np.zeros_like(img)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            neighbors = [img[i-1, j], img[i+1, j], img[i, j-1], img[i, j+1],
                         img[i-1, j-1], img[i-1, j+1], img[i+1, j-1], img[i+1, j+1]]
            pos = sum(n > 0 for n in neighbors)
            neg = sum(n < 0 for n in neighbors)
            if pos > 0 and neg > 0:
                result[i, j] = 255
    return result

hildreth = zero_crossing(log)

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1), plt.imshow(sobel, cmap='gray')
plt.title('Sobel'), plt.axis('off')

plt.subplot(2, 3, 2), plt.imshow(canny, cmap='gray')
plt.title('Canny'), plt.axis('off')

plt.subplot(2, 3, 3), plt.imshow(log, cmap='gray')
plt.title('LoG'), plt.axis('off')

plt.subplot(2, 3, 4), plt.imshow(dog, cmap='gray')
plt.title('DoG'), plt.axis('off')

plt.subplot(2, 3, 5), plt.imshow(hildreth, cmap='gray')
plt.title('Hildreth'), plt.axis('off')

plt.subplot(2, 3, 6), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.axis('off')

plt.show()
