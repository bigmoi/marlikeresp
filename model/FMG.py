import  torch
import torch.nn as nn
import random
import scipy.stats as stats
import numpy as np
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from model.diffloss import DiffLoss
class FMG(nn.Module):
    def __init__(self,batch_size=4,masks_rate=0.3,class_num=15,
                 encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=64,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,  # buffer size for token sequence,是一个填充的序列，暂时不用
                 diffloss_d=3,
                 diffloss_w=512,
                 diffusion_batch_mul=4,
                 num_sampling_steps='100',
                 grad_checkpointing=False,
                 seq_len=16, #序列长度，用于生成随机掩码# 莫名bug，无法改高，推测显存相关 #解决，应该与输入张量长度匹配

                 ):
        super(FMG, self).__init__()
        # --------------------------------------------------------------------------
        # VAE and patchify specifics 待修改.
        self.vae_embed_dim = vae_embed_dim

        self.seq_len=seq_len

        self.token_embed_dim = self.vae_embed_dim
        self.grad_checkpointing = grad_checkpointing

        self.z_pos_embed = nn.Parameter(torch.randn(1, seq_len, 64))  # 给变量z提供的位置嵌入
        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        # Label drop probability,to drop class embedding during training,improve robustness
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation, not used in FMG temprally
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        # self.buffer_size = buffer_size
        # self.encoder_pos_embed_learned = nn.Parameter(
        #     torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            # 标准vit的block
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.decoder_pos_embed_learned = nn.Parameter(
        #     torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))#不使用

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,#64
            z_channels=decoder_embed_dim,#512
            width=diffloss_w,#
            depth=diffloss_d,#
            num_sampling_steps=num_sampling_steps,#100
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul


    def initialize_weights(self):
            # parameters
            torch.nn.init.normal_(self.class_emb.weight, std=.02)
            torch.nn.init.normal_(self.fake_latent, std=.02)
            torch.nn.init.normal_(self.mask_token, std=.02)
            torch.nn.init.normal_(self.z_pos_embed, std=.02)  # 给z提供的位置嵌入的初始化
            # torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
            # torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
            # torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # mae编码器
    #要求输入x,mask,class_embedding
    # x：b*d
    def forward_mae_encoder(self, x, mask):
        x = self.z_proj(x)
        bsz,seq_len,embed_dim = x.shape



        # x = self.z_proj_ln(x)

        # dropping 这里取了反，即是原mask为1的地方丢弃，mask为0的地方保留
        # 仅拿 unmasked tokens 喂给 encoder
        x = x[(1 - mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks 实际编码部分.
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    #mae解码器
    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer =  mask

        # pad mask tokens
        # 由于 masked 仅仅是1个维度为 decoder embedding dim 的向量,
        # 因此要进行维度的扩展(在 batch 和 sequence 维度进行复制)
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)

        # 先全部初始化为 masked tokens, 而后把 encoder 的编码结果放到 unmasked 部分
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        x = x_after_pad
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        # 去掉 [cls] tokens 所对应的解码结果，因为没加，所以不使用
        # x = x[:, self.buffer_size:]
        # 一个额外的位置嵌入 不使用
        # x = x + self.diffusion_pos_embed_learned
        return x

    #输入shape 为 batch_size,joins*dim，frame,输出shape为batch_size,joins*dim，frame
    #输出为掩码张量
    def token_seq_mask_torch(self, batch_motion_seq):
        bsz,_,frame, = batch_motion_seq.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        mask_frame = int(frame * mask_rate)
        mask = torch.zeros(bsz,frame,device=batch_motion_seq.device).cuda()
        mask[ :,-mask_frame:] = 1

        return  mask

    # -----------------------拿来用的随机掩码---------------------------------------
    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))#生成一个0到seq_len-1的数组
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders
        # 随机掩码
    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    # --------------------------------------------------------------

    def _norm(self, batch_motion_seq):
        #将motion序列进行归一化
        return batch_motion_seq
    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape

        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, x):

        # class embed
        # class_embedding = self.class_emb(labels)
        x = x + self.z_pos_embed  #todo 添加位置嵌入 测试标记
        # mask = self.token_seq_mask(x)
        # 使用原版的随机掩码进行训练
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        gt_latents = x.clone().detach()

        # mae encoder
        x = self.forward_mae_encoder(x, mask) #8*-1*512

        # mae decoder
        z = self.forward_mae_decoder(x, mask)   #

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss


if __name__ == "__main__":
    #测试
    model = FMG(vae_embed_dim=64,seq_len=32)
    device= torch.device("cuda")
    model.to(device)
    data = torch.randn(8, 32, 64).to(device)

    for i in range(8):

        s1=model(data)
        print(s1)

