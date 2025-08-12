import os
import sys
import math
import pickle
import argparse
import time

import numpy as np
import torch_utils
import  torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from logger import create_logger

sys.path.append(os.getcwd())
from utils import *
from config import Config
from dataset_h36m import DatasetH36M
from dataset_humaneva import DatasetHumanEva
from model import VAE
import tqdm
# from motion_pred import *


def loss_function(X, Y_r, Y, mu, logvar):
    MSE = (Y_r - Y).pow(2).sum() / Y.shape[1]
    MSE_v = (X[-1] - Y_r[0]).pow(2).sum() / Y.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Y.shape[1]
    loss_r = MSE + cfg.lambda_v * MSE_v + cfg.beta * KLD
    return loss_r, np.array([loss_r.item(), MSE.item(), MSE_v.item(), KLD.item()])


def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'MSE', 'MSE_v', 'KLD']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size)
    for traj_np in tqdm.tqdm(generator,colour='#FF0000'):
        traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        traj = torch.tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        X = traj[:t_his]
        Y = traj[t_his:]
        Y_r, mu, logvar = model(X, Y)
        print("Y_r.shape",Y_r.shape)# [100, 64, 48]  l*b*projection
        loss, losses = loss_function(X, Y_r, Y, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += losses
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalar('vae_' + name, loss, epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_nsamp10')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    print("logger",logger)if logger is not None else print("logger None")
    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', t_his, t_pred, actions='all', use_vel=cfg.use_vel)
    if cfg.normalize_data:
        dataset.normalize_data()

    """model"""
    model = VAE.get_vae_model(cfg, dataset.traj_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = torch_utils.get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    #读取检查点
    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i)
            #保存检查点处
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:

                cp_path = cfg.vae_model_path % (i + 1)
                model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                pickle.dump(model_cp, open(cp_path, 'wb'))
                logger.info('save model to %s' % cp_path)
