import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


if __name__ == '__main__':
    data = loadmat('/media/zyj/data/zyj/遥感半监督多模态/│ú╝√╥ú╕╨╩²╛▌╝»/Augsburg/data_DSM.mat')
    print(data['hsi'].shape, data['lidar'].shape)
    hsi = data['hsi'].mean(-1)
    lidar = data['lidar'].mean(-1)
    label = data['label']
    print(label)

    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.imshow(hsi)

    plt.subplot(222)
    plt.imshow(lidar)

    plt.subplot(223)
    plt.imshow(label)
    plt.show()