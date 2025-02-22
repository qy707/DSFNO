# This file trains a downscaling FNO model.
# Author: Qidong Yang
# Date: 2022-08-26

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Adam import Adam
from Modules import *
from utilities import *
from evaluation import *


##### Configuration #####
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='NavierStokes', type=str)
parser.add_argument('--output_saving_path', default='./tmp/', type=str)
parser.add_argument('--n_channels', default=32, type=int)
parser.add_argument('--in_channel', default=1, type=int)
parser.add_argument('--n_residual_blocks', default=3, type=int)
parser.add_argument('--n_operator_blocks', default=2, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--eval_interval', default=20, type=int)
parser.add_argument('--x_res', default=16, type=int)
parser.add_argument('--y_res', default=32, type=int)
parser.add_argument('--modes', default=8, type=int)
parser.add_argument('--n_transforms', default=6, type=int)
parser.add_argument('--apply_constraint', default=1, type=int)

args = parser.parse_args()

data_path = '/data/' + args.data_folder + '/'
output_saving_path = args.output_saving_path
n_channels = args.n_channels
in_channel = args.in_channel
n_residual_blocks = args.n_residual_blocks
n_operator_blocks = args.n_operator_blocks
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
eval_interval = args.eval_interval
modes = args.modes
x_res = args.x_res
y_res = args.y_res
n_transforms = args.n_transforms

if args.apply_constraint == 1:
    apply_constraint = True
else:
    apply_constraint = False

print(' ', flush=True)
print('Experiment Configuration', flush=True)
print('n_channels: ', n_channels, flush=True)
print('in_channel: ', in_channel, flush=True)
print('apply_constraint: ', apply_constraint, flush=True)
print('n_residual_blocks: ', n_residual_blocks, flush=True)
print('n_operator_blocks: ', n_operator_blocks, flush=True)
print('lr: ', lr, flush=True)
print('epochs: ', epochs, flush=True)
print('batch_size: ', batch_size, flush=True)
print('eval_interval: ', eval_interval, flush=True)
print('modes: ', modes, flush=True)
print('x_res: ', x_res, flush=True)
print('y_res: ', y_res, flush=True)
print('n_transforms: ', n_transforms, flush=True)
print('data_path: ', data_path, flush=True)
print('output_saving_path: ', output_saving_path, flush=True)


##### Load Data #####
train_path = data_path + 'train/'
valid_path = data_path + 'val/'
test_path = data_path + 'test/'

train_x = torch.load(train_path + 'data_' + str(x_res) + '.pt').squeeze().unsqueeze(-1)
valid_x = torch.load(valid_path + 'data_' + str(x_res) + '.pt').squeeze().unsqueeze(-1)
test_x = torch.load(test_path + 'data_' + str(x_res) + '.pt').squeeze().unsqueeze(-1)

train_y = torch.load(train_path + 'data_' + str(y_res) + '.pt').squeeze().unsqueeze(-1)
valid_y = torch.load(valid_path + 'data_' + str(y_res) + '.pt').squeeze().unsqueeze(-1)

# process inputs
normalizer = MaxMinNormalizer(train_y)
train_x_n = normalizer.encode(train_x)
valid_x_n = normalizer.encode(valid_x)
test_x_n = normalizer.encode(test_x)

# process outputs
test_y_16 = torch.load(test_path + 'data_16.pt').squeeze().unsqueeze(-1)
test_y_32 = torch.load(test_path + 'data_32.pt').squeeze().unsqueeze(-1)
test_y_64 = torch.load(test_path + 'data_64.pt').squeeze().unsqueeze(-1)

# get data loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x_n, train_y), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x_n, valid_y), batch_size=batch_size, shuffle=False)

# get data augmentor
augmentor = DataAugmentor()


##### Define Model #####
model = DSFNO(in_channel, n_channels, n_residual_blocks, n_operator_blocks, modes, apply_constraint).cuda()

nn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Parameter Number: ', nn_params, flush=True)
print(' ', flush=True)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

myloss = LpLoss(size_average=False)


###### Training ######
normalizer.cuda()

train_lps = []
train_mses = []
valid_lps = []
valid_mses = []
test_mses_16 = []
test_mses_32 = []
test_mses_64 = []

for epoch in range(epochs):

    model.train()
    train_lp = 0
    train_mse = 0

    for x, y in train_loader:

        x, y = x.cuda(), y.cuda()
        x, y = augmentor.transform(x, y, n_transforms)

        optimizer.zero_grad()

        out = model(x, int(y_res/x_res))
        out = normalizer.decode(out)

        lp = myloss(out, y)
        lp.backward()

        optimizer.step()

        train_lp = train_lp + lp.item()
        train_mse = train_mse + F.mse_loss(out, y, reduction='mean').item() * y.size(0)

    train_lp = train_lp / len(train_y)
    train_mse = train_mse / len(train_y)

    model.eval()
    valid_lp = 0
    valid_mse = 0

    with torch.no_grad():
        for x, y in valid_loader:

            x, y = x.cuda(), y.cuda()

            out = model(x, int(y_res/x_res))
            out = normalizer.decode(out)

            valid_lp = valid_lp + myloss(out, y).item()
            valid_mse = valid_mse + F.mse_loss(out, y, reduction='mean').item() * y.size(0)

    valid_lp = valid_lp / len(valid_y)
    valid_mse = valid_mse / len(valid_y)

    train_lps.append(train_lp)
    valid_lps.append(valid_lp)
    train_mses.append(train_mse)
    valid_mses.append(valid_mse)

    print('Epoch: %d train_loss[%.4f] train_mse[%.4f] valid_loss[%.4f] valid_mse[%.4f]' % (epoch + 1, train_lp, train_mse, valid_lp, valid_mse), flush=True)
    print(' ', flush=True)

    if (epoch + 1) % eval_interval == 0 or epoch == 0:

        test_preds_16 = GetTestPreds_batch(model, test_x_n, normalizer, upsample_factor=int(16/x_res), batch_size=batch_size)
        test_preds_32 = GetTestPreds_batch(model, test_x_n, normalizer, upsample_factor=int(32/x_res), batch_size=batch_size)
        test_preds_64 = GetTestPreds_batch(model, test_x_n, normalizer, upsample_factor=int(64/x_res), batch_size=batch_size)

        test_lp1_16, test_lp2_16, test_mse_16, test_mae_16, test_psnr_16, test_ssim_16 = GetTestMetrics(test_preds_16, test_y_16)
        test_lp1_32, test_lp2_32, test_mse_32, test_mae_32, test_psnr_32, test_ssim_32 = GetTestMetrics(test_preds_32, test_y_32)
        test_lp1_64, test_lp2_64, test_mse_64, test_mae_64, test_psnr_64, test_ssim_64 = GetTestMetrics(test_preds_64, test_y_64)

        print('Resolution %d-->16 Test Report: lp1[%.4f] lp2[%.4f] mse[%.4f] mae[%.4f] psnr[%.4f] ssim[%.4f]' % (x_res, test_lp1_16, test_lp2_16, test_mse_16, test_mae_16, test_psnr_16, test_ssim_16), flush=True)
        print('Resolution %d-->32 Test Report: lp1[%.4f] lp2[%.4f] mse[%.4f] mae[%.4f] psnr[%.4f] ssim[%.4f]' % (x_res, test_lp1_32, test_lp2_32, test_mse_32, test_mae_32, test_psnr_32, test_ssim_32), flush=True)
        print('Resolution %d-->64 Test Report: lp1[%.4f] lp2[%.4f] mse[%.4f] mae[%.4f] psnr[%.4f] ssim[%.4f]' % (x_res, test_lp1_64, test_lp2_64, test_mse_64, test_mae_64, test_psnr_64, test_ssim_64), flush=True)
        print(' ', flush=True)

        torch.save(test_preds_16[5000, :, :, 0], output_saving_path + 'test_preds_res_16_epoch_' + str(int(epoch + 1)) + '.pt')
        test_mses_16.append(test_mse_16)

        torch.save(test_preds_32[5000, :, :, 0], output_saving_path + 'test_preds_res_32_epoch_' + str(int(epoch + 1)) + '.pt')
        test_mses_32.append(test_mse_32)

        torch.save(test_preds_64[5000, :, :, 0], output_saving_path + 'test_preds_res_64_epoch_' + str(int(epoch + 1)) + '.pt')
        test_mses_64.append(test_mse_64)

        preds = [test_preds_16[5000, :, :, 0], test_preds_32[5000, :, :, 0], test_preds_64[5000, :, :, 0]]
        targets = [test_y_16[5000, :, :, 0], test_y_32[5000, :, :, 0], test_y_64[5000, :, :, 0]]

        PlotPredComparison(preds=preds, targets=targets, epoch=epoch + 1, output_path=output_saving_path)


##### Save #####
train_lps = np.array(train_lps)
np.save(output_saving_path + 'train_lps.npy', train_lps)

valid_lps = np.array(valid_lps)
np.save(output_saving_path + 'valid_lps.npy', valid_lps)

train_mses = np.array(train_mses)
np.save(output_saving_path + 'train_mses.npy', train_mses)

valid_mses = np.array(valid_mses)
np.save(output_saving_path + 'valid_mses.npy', valid_mses)

test_mses_16 = np.array(test_mses_16)
np.save(output_saving_path + 'test_mses_16.npy', test_mses_16)

test_mses_32 = np.array(test_mses_32)
np.save(output_saving_path + 'test_mses_32.npy', test_mses_32)

test_mses_64 = np.array(test_mses_64)
np.save(output_saving_path + 'test_mses_64.npy', test_mses_64)

torch.save(model, output_saving_path + 'model.pt')


##### Plotting #####
PlotTrainingCurve(train_lps, valid_lps, output_saving_path)
PlotMSE([train_mses, valid_mses], [x_res, y_res], [test_mses_16, test_mses_32, test_mses_64], [[x_res, 16], [x_res, 32], [x_res, 64]], eval_interval, output_saving_path)