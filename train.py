import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import Dataset
import data_loader
from net import nets, net_v1
from utils import config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_loss(preds, target, mask, epoch):
    def loss_super(pred, target, mask):
        pred = pred.view(pred.shape[0], num_class, -1)
        target = target.view(target.shape[0], -1)
        mask = mask.view(mask.shape[0], -1)
        idxs = torch.where(mask > 0)
        pred = pred[idxs[0], :, idxs[1]]
        target = target[idxs]
        loss = criterion(pred, target).mean()
        return loss

    def loss_unsuper(inputs1, inputs2, mask):
        inputs1 = inputs1.view(inputs1.shape[0], num_class, -1)
        inputs2 = inputs2.view(inputs2.shape[0], num_class, -1)
        inputs1 = torch.softmax(inputs1, dim=1)
        inputs2 = torch.softmax(inputs2, dim=1)
        mask = mask.view(mask.shape[0], -1)
        idxs = torch.where(mask <= 0)

        inputs1 = inputs1[idxs[0], :, idxs[1]]
        inputs2 = inputs2[idxs[0], :, idxs[1]]
        return criterion_mse(inputs1, inputs2)

    hsi_pred, lidar_pred, pred_out = preds
    loss_hsi = loss_super(hsi_pred, target, mask)
    loss_lidar = loss_super(lidar_pred, target, mask)
    loss = loss_super(pred_out, target, mask)

    loss_mse_hsi_lidar = loss_unsuper(hsi_pred, lidar_pred, mask)
    loss_mse_hsi_out = loss_unsuper(hsi_pred, pred_out, mask)
    loss_mse_lidar_out = loss_unsuper(lidar_pred, pred_out, mask)

    rate = epoch / epochs + 1.
    rate1 = epoch / epochs
    return rate * (loss + loss_hsi + loss_lidar) + rate1 * (loss_mse_lidar_out + loss_mse_hsi_out + loss_mse_hsi_lidar)


def fit():
    best_loss = 1000
    for epoch in range(epochs):
        dt_size = len(train_data_loader.dataset)
        dt_size_val = len(test_data_loader.dataset)
        epoch_loss = 0
        step = 0
        pbar = tqdm(total=dt_size // batch_size,
                    desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict,
                    mininterval=0.3)
        for x_hsi, x_lidar, mask, y in train_data_loader:
            x_hsi = x_hsi.to(device)
            x_lidar = x_lidar.to(device)
            mask = mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x_hsi, x_lidar)
            loss = get_loss(pred, y, mask, epoch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{
                'train_loss': epoch_loss / (step + 1)
            })
            pbar.update(1)
            step += 1
        pbar.close()
        pbar = tqdm(total=dt_size_val // batch_size,
                    desc=f'Val_Epoch {epoch + 1}/{epochs}', postfix=dict,
                    mininterval=0.3)
        epoch_loss_val = 0
        step_val = 0
        for x_hsi, x_lidar, mask, y in test_data_loader:
            x_hsi = x_hsi.to(device)
            x_lidar = x_lidar.to(device)
            mask = mask.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(x_hsi, x_lidar)
            loss = get_loss(pred, y, mask, epoch)
            epoch_loss_val += loss.item()
            pbar.set_postfix(**{'val_loss': epoch_loss_val / (step_val + 1)})

            pbar.update(1)
            step_val += 1
        pbar.close()
        if best_loss > epoch_loss_val / step_val:
            best_loss = epoch_loss_val / step_val
            torch.save(
                {
                    'epoch': epoch + 1,
                    'optimizer_dict': optimizer.state_dict(),
                    'model_dict': model.state_dict(),
                }, save_weights_path + 'weights.pth'
            )


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 16
    use_data = 'Augsburg'
    config = config.configs[use_data]
    num_class = config.num_class

    epochs = 30
    save_weights_path = './save_weights_' + use_data + '/'
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    train_files, test_files = data_loader.get_data_path(use_data)

    model = net_v1.build(input_shape=(32, 32), num_classes=config.num_class, in_chans_hsi=config.input_hsi_channel,
                         in_chans_lidar=config.input_lidar_channel).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_mse = torch.nn.MSELoss()

    train_data_loader = DataLoader(Dataset(train_files), batch_size=batch_size,
                                   num_workers=4, shuffle=True)
    test_data_loader = DataLoader(Dataset(test_files), batch_size=batch_size,
                                  num_workers=4, shuffle=False)
    fit()
