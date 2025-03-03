import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from utils import config
from utils.crop import nor
from scipy.io import loadmat
from net import nets, net_v1
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
crop_size = 32

colormap = np.array([[0, 0, 205],
                     [0, 8, 255],
                     [0, 77, 255],
                     [0, 145, 255],
                     [0, 212, 255],
                     [41, 255, 206],
                     [96, 255, 151],
                     [151, 255, 96],
                     [206, 255, 41],
                     [255, 230, 0],
                     [255, 167, 0],
                     [255, 104, 0],
                     [255, 41, 0],
                     [205, 0, 0],
                     [128, 0, 0],
                     [128, 0, 0]])


def get_inputs_data(use_data):
    if use_data == 'Houston2013':
        img1 = loadmat('./data_begin/Houston2013/HSI.mat')['HSI']
        img2 = loadmat('./data_begin/Houston2013/LiDAR_1.mat')['LiDAR']
    elif use_data == 'MUFFL':
        img1 = loadmat('./data_begin/MUFFL/hsi.mat')['HSI']

        img2 = loadmat('./data_begin/MUFFL/lidar.mat')['lidar']
    elif use_data == 'Trento':
        img1 = loadmat('./data_begin/Trento/HSI.mat')['HSI']

        img2 = loadmat('./data_begin/Trento/LiDAR.mat')['LiDAR']
    elif use_data == 'Augsburg':
        img1 = loadmat('./data_begin/Augsburg/data_hsi.mat')['data']

        img2 = loadmat('./data_begin/Augsburg/data_sar.mat')['data']
    if len(img1.shape) == 2:
        img1 = np.pad(nor(img1), (crop_size // 2, crop_size // 2), constant_values=0)[..., None]
    else:
        img1 = np.stack([np.pad(nor(img1[..., i]), (crop_size // 2, crop_size // 2), constant_values=0)
                         for i in range(img1.shape[-1])], axis=-1)
    if len(img2.shape) == 2:
        img2 = np.pad(nor(img2), (crop_size // 2, crop_size // 2), constant_values=0)[..., None]
    else:
        img2 = np.stack([np.pad(nor(img2[..., i]), (crop_size // 2, crop_size // 2), constant_values=0)
                         for i in range(img2.shape[-1])], axis=-1)
    return img1, img2


if __name__ == '__main__':
    device = 'cuda'
    use_data = 'MUFFL'
    config = config.configs[use_data]
    num_class = config.num_class
    save_weights_path = './save_weights_' + use_data + '/weights.pth'

    model = net_v1.build(input_shape=(32, 32), num_classes=config.num_class, in_chans_hsi=config.input_hsi_channel,
                         in_chans_lidar=config.input_lidar_channel).to(device)

    key = model.load_state_dict(torch.load(save_weights_path, map_location=device)['model_dict'])
    model.eval()
    print(key)

    hsi, lidar = get_inputs_data(use_data)

    pad_x = crop_size - hsi.shape[0] % crop_size
    pad_y = crop_size - hsi.shape[1] % crop_size

    x = hsi.shape[0] + pad_x % crop_size
    y = hsi.shape[1] + pad_y % crop_size

    hsi_pad = np.zeros(shape=(x, y, hsi.shape[-1] if len(hsi.shape) > 2 else 1))
    hsi_pad[pad_x % crop_size:, pad_y % crop_size:] = hsi
    lidar_pad = np.zeros(shape=(x, y, lidar.shape[-1] if len(lidar.shape) > 2 else 1))
    lidar_pad[pad_x % crop_size:, pad_y % crop_size:] = lidar

    outs = np.zeros(shape=(x, y))
    for i in tqdm.tqdm(range(crop_size // 2, x - crop_size // 2 + 1, crop_size)):
        for j in range(crop_size // 2, y - crop_size // 2 + 1, crop_size):
            crop_hsi = hsi_pad[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2]
            crop_lidar = lidar_pad[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2]

            crop_hsi = torch.from_numpy(crop_hsi.transpose(2, 0, 1)).type(torch.FloatTensor)
            crop_lidar = torch.from_numpy(crop_lidar.transpose(2, 0, 1)).type(torch.FloatTensor)
            with torch.no_grad():
                crop_hsi = crop_hsi.to(device)
                crop_lidar = crop_lidar.to(device)
                *_, pre = model(crop_hsi[None], crop_lidar[None])
                pre = torch.softmax(pre, dim=1)[0]
            pre = pre.cpu().numpy()
            pre = np.argmax(pre, 0)
            outs[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2] = pre
    outs = outs[pad_x % crop_size:, pad_y % crop_size:] + 1
    outs = outs.astype(np.uint8)
    x, y = outs.shape
    seg_out = colormap[outs.flatten()].reshape((x, y, 3))
    seg_out = seg_out[crop_size // 2: -crop_size // 2, crop_size // 2: -crop_size // 2]
    Image.fromarray(seg_out.astype(np.uint8)).save('./result/' + use_data + '.png')
    plt.figure(figsize=(12, 12))
    plt.imshow(seg_out)
    plt.show()
