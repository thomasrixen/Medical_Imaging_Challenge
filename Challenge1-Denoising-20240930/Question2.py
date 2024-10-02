import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from question1 import calculate_snr, gaussian_noise

def gaussianKernel(size, fwhm=3):
    """ gaussianKernel returns a gaussian kernel of size (sizew x size)
    INPUTS :
        - size: length of a side of the square
        - fwhm: full-width-half-maximum (can be thought of as an effective radius)
    OUTPUT:
        - gaussian kernel with above specifications
    """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    kernel = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

    return kernel / np.sum(kernel)


def linearFilter(f, w):
    """ linearFilter implements the convolution of 2D image by a 2D kernel
    INPUTS :
        - f: image to be filtered
        - w: kernel
    OUTPUT:
        - result of the convultion between f and w
    """
    print(f.shape)

    # Get image and kernel dimensions
    image_h, image_w = f.shape
    kernel_h, kernel_w = w.shape

    # Calculate padding for the image (to handle border issues)
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the image with zeros
    padded_image = np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Initialize output image
    filtered_image = np.zeros_like(f)

    # Convolution operation
    for i in range(image_h):
        for j in range(image_w):
            # Perform the convolution using a nested loop over the kernel
            for k in range(kernel_h):
                for l in range(kernel_w):
                    filtered_image[i, j] += w[k, l] * padded_image[i + k, j + l]

    return filtered_image


ORIGINAL_IMAGE_PATH = '/home/thomas/Documents/UCL/Master1/Q1/LGBIO2050-Medical_Imaging/Challenges/Challenge1-Denoising-20240930/imgs/ChestXRay.jpeg'
NOISY_IMAGE_PATH = '/home/thomas/Documents/UCL/Master1/Q1/LGBIO2050-Medical_Imaging/Challenges/Challenge1-Denoising-20240930/imgs/NoisyChestXRay.jpeg'

original_image = mpimg.imread(ORIGINAL_IMAGE_PATH)
noisy_image = mpimg.imread(NOISY_IMAGE_PATH)


# Normalize images if needed
if noisy_image.max() <= 1:
    noisy_image = noisy_image * 255
if original_image.max() <= 1:
    original_image = original_image * 255

noisy_image = noisy_image.astype(np.float32)
original_image = original_image.astype(np.float32)

# 2. Generate a Gaussian kernel
kernel_size = 5  # Example size for Gaussian kernel
fwhm = 3  # Full-width half-maximum
gaussian_kernel = gaussianKernel(kernel_size, fwhm)

# 3. Apply the linear filter (Gaussian blur) to the noisy image
denoised_image = linearFilter(noisy_image, gaussian_kernel)

# 4. Compute SNR before and after filtering
snr_before = calculate_snr(original_image, noisy_image)
snr_after = calculate_snr(original_image, denoised_image)

print(f"SNR before filtering: {snr_before:.2f} dB")
print(f"SNR after filtering: {snr_after:.2f} dB")

# 5. Plot the original, noisy, and denoised images
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_image.astype(np.uint8), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image.astype(np.uint8), cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(denoised_image.astype(np.uint8), cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.tight_layout()
plt.show()