import yaml
import os
from pathlib import Path

class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = os.path.join(Path(__file__).parent,'cfg/%s.yml' % cfg_id)
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = '/tmp' if test else 'results'

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg.get('batch_size', 8)
        self.normalize_data = cfg.get('normalize_data', False)
        self.save_model_interval = cfg.get('save_model_interval', 20)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.use_vel = cfg.get('use_vel', False)

        # vae
        self.nz = cfg['nz']
        self.beta = cfg['beta']
        self.lambda_v = cfg.get('lambda_v', 0)
        self.vae_lr = cfg['vae_lr']
        self.vae_specs = cfg.get('vae_specs', dict())
        self.num_vae_epoch = cfg['num_vae_epoch']
        self.num_vae_epoch_fix = cfg.get('num_vae_epoch_fix', self.num_vae_epoch)
        self.num_vae_data_sample = cfg['num_vae_data_sample']
        self.vae_model_path = os.path.join(self.model_dir , 'vae_%04d.p')

        #mae params
        self.mask_ratio_min = cfg['mask_ratio_min']
        self.grad_clip = cfg['grad_clip']
        self.attn_dropout = cfg['attn_dropout']
        self.proj_dropout = cfg['proj_dropout']
        self.buffer_size = cfg['buffer_size']

        self.diffloss_d = cfg['diffloss_d']
        self.diffloss_w = cfg['diffloss_w']
        self.num_sampling_steps = cfg['num_sampling_steps']
        self.diffusion_batch_mul = cfg['diffusion_batch_mul']
        self.temperature = cfg['temperature']

        self.weight_decay = cfg['weight_decay']
        self.grad_checkpointing = cfg.get('grad_checkpointing', False)
        self.lr = cfg['lr']
        self.blr = cfg['blr']
        self.min_lr = cfg['min_lr']
        self.lr_schedule = cfg['lr_schedule']
        self.warmup_epochs = cfg['warmup_epochs']
        self.ema_rate = cfg['ema_rate']

    def addparams(self, args):
        # args_dict = vars(args) if hasattr(args, '__dict__') else args
        # for i in args_dict:
        #     if hasattr(self, args_dict[i]):
        #         pass
        #     else:
        #         setattr(self, i, args_dict[i])
        if hasattr(args, '__dict__'):
            # 如果是 Namespace 对象，获取其 __dict__ 属性
            args_dict = vars(args)
            for key, value in args_dict.items():
                if hasattr(self, key):
                    pass
                else:
                    setattr(self, key, value)
        elif isinstance(args, dict):
            # 如果是字典，直接遍历
            for key, value in args.items():
                if hasattr(self, key):
                    setattr(self, key, value)