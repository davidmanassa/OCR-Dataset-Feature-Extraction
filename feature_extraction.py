import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import filters
from skimage.filters import gabor_kernel

def countBlackPixels(img_lst):
    count = 0
    for x in img_lst:
        if x != 255:
            count += 1
    return count

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(7, 7))
plt.gray()

fig.suptitle('Gabor filter kernels', fontsize=12)

i = 0
for a in range(4):
    for b in range(4):
        axes[a][b].imshow(np.real(kernels[i]), interpolation='nearest', cmap='gray')
        axes[a][b].set_xticks([])
        axes[a][b].set_yticks([])
        i += 1

plt.show()


def countPixels(img_mat):
    count = 0
    for i, x in enumerate(img_mat):
        for j, y in enumerate(x):
            if y >= 0.00000002:
                count+=1
    return count
"""
Obtemos as features dando uma imagem
"""
def getFeatures(img):
    img_matrix = [img[i : i + 50] for i in range(0, len(img), 50)]
    
    blackPixels = [np.double(countBlackPixels(img))]

    # prewitt kernel
    pre_hor = filters.prewitt_h(img_matrix)
    pre_ver = filters.prewitt_v(img_matrix)

    blackPixels.append(countPixels(pre_hor))
    blackPixels.append(countPixels(pre_ver))

    tmp = compute_feats(img_matrix, kernels)
    gaborFeatures = []
    for f in tmp:
        gaborFeatures.extend(f)
    return blackPixels + gaborFeatures

"""
Obtemos as features e labels dando uma lista de imagens e a label
"""
def getFeaturesLst(img_lst, label):
    features = []
    labels = []
    for img in img_lst:
        features.append(getFeatures(img))
        labels.append(label)
    return features, labels