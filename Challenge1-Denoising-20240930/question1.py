import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def gaussian_noise(mean, std_dev, shape):
    return np.random.normal(mean, std_dev, shape)


# Signal-to-Noise Ratio (SNR) in dB, from the slides
def calculate_snr(original, noisy):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - noisy) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# Write here the relative of full path to the /imgs/ChestXRay.jpeg
ORIGINAL_IMAGE_PATH = '/home/thomas/Documents/UCL/Master1/Q1/LGBIO2050-Medical_Imaging/Challenges/Challenge1-Denoising-20240930/imgs/ChestXRay.jpeg'
NOISY_IMAGE_SAVE_PATH = '/home/thomas/Documents/UCL/Master1/Q1/LGBIO2050-Medical_Imaging/Challenges/Challenge1-Denoising-20240930/imgs/NoisyChestXRay.jpeg'

original_image = mpimg.imread(ORIGINAL_IMAGE_PATH)

if original_image.max() <= 1:
    original_image = original_image * 255
original_image = original_image.astype(np.float32)

mean = 0
std_dev = 5

noise = gaussian_noise(mean, std_dev, original_image.shape)
noisy_image = original_image + noise

noisy_image = np.clip(noisy_image, 0, 255)

snr_value = calculate_snr(original_image, noisy_image)
print(f'SNR: {snr_value:.2f} dB')

plt.imsave(NOISY_IMAGE_SAVE_PATH, noisy_image.astype(np.uint8), cmap='gray')

plt.figure(figsize=(18, 6))

# plot the original image
plt.subplot(1, 3, 1)
plt.imshow(original_image.astype(np.uint8), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# plot the noisy image
plt.subplot(1, 3, 2)
plt.imshow(noisy_image.astype(np.uint8), cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

# The difference between both picture
plt.subplot(1, 3, 3)
plt.imshow(np.abs(original_image - noisy_image).astype(np.uint8), cmap='gray')
plt.title('Difference (|Original - Noisy|)')
plt.axis('off')

plt.tight_layout()
#plt.show()
