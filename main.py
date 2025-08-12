
from utils.dataset_h36m import DatasetH36M
from utils.dataset_humaneva import DatasetHumanEva
from model.VAE import get_vae_model
from model.FMG import FMG

import torch
import argparse
from utils.config import Config
import numpy as np
def main(args):
    device = torch.device(args.device)
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_cls = DatasetH36M if args.dataset == 'h36m' else DatasetHumanEva
    if args.dataset == 'h36m':
        dataset = dataset_cls('train', args.t_his, args.t_pred, actions='all', use_vel=args.use_vel)
        if args.normalize_data:
            dataset.normalize_data()
    elif args.dataset == 'humaneva':
        1
        # dataset = dataset_cls('train', t_his, t_pred, actions='all', use_vel=cfg.use_vel)
        # if cfg.normalize_data:
        #     dataset.normalize_data()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    vae= get_vae_model(args, dataset.traj_dim)
    model = FMG(batch_size=args.batch_size, masks_rate=0.7, class_num=15,
                encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16,
                mlp_ratio=4.,
                norm_layer=torch.nn.LayerNorm,
                vae_embed_dim=64,
                mask_ratio_min=0.7,
                label_drop_prob=0.1,
                attn_dropout=0.1,
                proj_dropout=0.1,
                buffer_size=64,  # buffer size for token sequence,是一个填充的序列，暂时不用 #测试用.
                diffloss_d=args.diffloss_d,
                diffloss_w=args.diffloss_w,
                diffusion_batch_mul=args.diffusion_batch_mul,
                num_sampling_steps=args.num_sampling_steps,
                grad_checkpointing=True,
                seq_len=32,
                )

    print(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--device_index', type=int, default=0)
    cfg = parser.parse_args()

    args=Config('h36m_nsamp10')
    args.addparams(cfg)
    print('测试是否成功加入变量',args.device)

    main(args)