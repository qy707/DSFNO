# This file contains functions to evaluate model performance.
# Author: Qidong Yang
# Date: 2022-07-22


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchmetrics.functional as MF

from Modules import *
from utilities import *


def GetTestPreds_batch(model, test_x_n, y_normalizer, upsample_factor, input_factor=True, batch_size=200):

    # Compute test prediction
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x_n), batch_size=batch_size, shuffle=False)

    x_size = test_x_n.shape[1]
    y_size = test_x_n.shape[2]
    n_test = test_x_n.shape[0]

    preds = torch.ones((n_test, x_size * upsample_factor, y_size * upsample_factor, 1))

    idx = 0
    with torch.no_grad():
        for x in test_loader:

            x = x[0].cuda()

            if input_factor:
                pred = model(x, upsample_factor)
            else:
                pred = model(x)

            pred = y_normalizer.decode(pred)

            preds[np.arange(batch_size) + idx] = pred.cpu()

            idx = idx + batch_size

    return preds


def GetTestMetrics(test_preds, test_y, padding_factor=8 / 128):
    
    # test_preds: (n_test, x_size, y_size, 1)

    # Compute test metrics
    Lp1 = LpLoss(d=2, p=1)
    Lp2 = LpLoss(d=2, p=2)

    test_lp1 = Lp1(test_preds, test_y).item()
    test_lp2 = Lp2(test_preds, test_y).item()
    test_mse = F.mse_loss(test_preds, test_y, reduction='mean').item()
    test_mae = MF.mean_absolute_error(test_preds, test_y).item()
    test_psnr = MF.peak_signal_noise_ratio(test_preds, test_y).item()
    test_ssim = MF.structural_similarity_index_measure(test_preds.permute(0, 3, 1, 2), test_y.permute(0, 3, 1, 2)).item()


    #print('Resolution %d Test Report: lp1[%.3f] lp2[%.3f] mse[%.3f]' % (res, test_lp1, test_lp2, test_mse), flush=True)

    return test_lp1, test_lp2, test_mse, test_mae, test_psnr, test_ssim


def PlotPredComparison(preds, targets, epoch, output_path):

    n_row = len(preds)

    fig, axs = plt.subplots(n_row, 3, figsize=(18, 5 * n_row), squeeze=False)

    for i in range(n_row):

        cb = axs[i, 0].pcolormesh(preds[i], vmin=-1.5, vmax=1.5)
        plt.colorbar(cb, ax=axs[i, 0])
        axs[i, 0].set_title('Epoch ' + str(epoch) + ': Prediction')

        cb = axs[i, 1].pcolormesh(targets[i], vmin=-1.5, vmax=1.5)
        plt.colorbar(cb, ax=axs[i, 1])
        axs[i, 1].set_title('Epoch ' + str(epoch) + ': Target')

        cb = axs[i, 2].pcolormesh(targets[i] - preds[i], cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        plt.colorbar(cb, ax=axs[i, 2])
        axs[i, 2].set_title('Epoch ' + str(epoch) + ': Target - Prediction')

    plt.savefig(output_path + 'pred_comparison_' + str(epoch) + '.png')
    plt.close()


def PlotTrainingCurve(train_loss, valid_loss, output_path, loss_type=None):

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(np.arange(len(train_loss)) + 1, train_loss, label='Training Loss')
    axs.plot(np.arange(len(valid_loss)) + 1, valid_loss, label='Validation Loss')
    axs.legend()
    axs.grid()

    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    #axs.set_yscale('log')
    #axs.set_ylim(0.01, 0.2)

    if loss_type == None:
        title_name = 'Training Curve'
        file_name = 'training_curve.png'
    else:
        title_name = loss_type + ' Training Curve'
        file_name = loss_type.replace(' ', '_') + '_training_curve.png'

    axs.set_title(title_name)

    plt.savefig(output_path + file_name)
    plt.close()


def PlotMSE(learn_mses, learn_res, test_mses, test_res, eval_interval, output_path):
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        
    epochs = len(learn_mses[0])
    axs.plot(np.arange(1, epochs + 1), learn_mses[0], label='Train: ' + str(learn_res[0]) + ' -> ' + str(learn_res[-1]))
    axs.plot(np.arange(1, epochs + 1), learn_mses[1], label='Valid: ' + str(learn_res[0]) + ' -> ' + str(learn_res[-1]))

    for i in range(len(test_mses)):

        loss_len = len(test_mses[i]) - 1

        x_part_1 = np.array([1])
        x_part_2 = np.arange(1, loss_len + 1) * eval_interval
        x_axis = np.concatenate([x_part_1, x_part_2])

        axs.plot(x_axis, test_mses[i], label='Test: ' + str(test_res[i][0]) + ' -> ' + str(test_res[i][-1]))
    
    axs.legend()
    axs.grid()
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Mean Squared Error')
    axs.set_ylim(0, 0.04)

    axs.set_title('Mean Squared Error Curve')

    plt.savefig(output_path + 'mse_curve.png')
    plt.close()