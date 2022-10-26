from timm.models import layers
from timm.models.helpers import default_cfg_for_features
import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, HybridEmbed
import math

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

__all__ = [
    'deit_tiny_cfged_patch16_224', 
    'deit_small_cfged_patch16_224',
    'deit_base_cfged_patch16_224'
]

################################################################################################################
def selector(x, attn, n_tokens, cutoff=0.15):
    B, N, C = x.shape
    if n_tokens == N-1:
        return x
    elif n_tokens == 0:
        return x[:, 0].reshape(B, 1, C)
    else:
        # attention scores
        cls_attn_weight = attn[:, :, 1:, 0].mean(-1).reshape(B, -1, 1)
        cls_attn = attn[:, :, 0, 1:].mul(cls_attn_weight).mean(1)
        img_attn_weight = attn[:, :, 0, 0].reshape(B, -1, 1)
        img_attn = attn[:, :, 1:, 1:].mean(2).mul(img_attn_weight).mean(1)
        token_attn = cls_attn.add(img_attn)

        cls_token = x[:, 0].reshape(B, 1, C)
        img_token = x[:, 1:].float().permute(0, 2, 1)
        # FFT for torch 1.8
        if IS_HIGH_VERSION:
            fft_token = torch.fft.fft(img_token.detach(), dim=-1)
            fft_token = torch.stack([fft_token.real, fft_token.imag], -1)
            pha = torch.atan2(fft_token[:,:,:,1], fft_token[:,:,:,0])
            fft_token = torch.sqrt(fft_token[:,:,:,0]**2 + fft_token[:,:,:,1]**2)
        # FFT for torch 1.7
        else:
            fft_token = torch.rfft(img_token.detach(), signal_ndim=1, onesided=False)
            pha = torch.atan2(fft_token[:,:,:,1], fft_token[:,:,:,0])
            fft_token = torch.sqrt(fft_token[:,:,:,0]**2 + fft_token[:,:,:,1]**2)

        # gaussian filter
        d0 = (fft_token.shape[-1] * cutoff / 2) ** 2
        m0 = (fft_token.shape[-1] - 1) / 2.
        x_grid = torch.arange(fft_token.shape[-1]).to(x.device)
        kernel = 1 - torch.exp(-((x_grid - m0)**2.) / (2*d0))

        fft_token.mul_(kernel)
        a1 = torch.cos(pha) * fft_token
        a2 = torch.sin(pha) * fft_token
        fft_src_ = torch.cat([a1.unsqueeze(-1),a2.unsqueeze(-1)],dim=-1)
        # IFFT for torch 1.8
        if IS_HIGH_VERSION:
            fft_src_ = torch.complex(fft_src_[..., 0], fft_src_[..., 1])
            fft_token = torch.fft.ifft(fft_src_, dim=-1)
        # IFFT for torch 1.7
        else:
            fft_token = torch.irfft(fft_src_, signal_ndim=1, onesided=False)
        
        token_lfe = fft_token.abs().sum(1)/img_token.detach().abs().sum(1).mean(-1).unsqueeze(-1)

        scores = token_lfe.reshape(B, N-1).mul(token_attn)

        _, idx = torch.topk(scores, n_tokens, dim=1, largest=True, sorted=False)
        sav_token = torch.gather(x[:, 1:], dim=1, index=idx.unsqueeze(-1).expand(-1, -1, C))
        
        return torch.cat((cls_token, sav_token), dim=1)

class Attention_cfged(nn.Module):
    def __init__(self, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 dim_msa=None, n_token=196):
        super().__init__()
        dim_q, dim_k, dim_v, dim_proj = dim_msa[0], dim_msa[1], dim_msa[2], dim_msa[3]
        self.C_q, self.C_k, self.C_v = dim_q[1], dim_k[1], dim_v[1]
        if self.C_q % num_heads != 0 or self.C_k % num_heads != 0 or self.C_v % num_heads != 0:
            raise ValueError(f'channels of "q, k, v" must be divisible by num_heads! n_head:{num_heads}, d_q:{self.C_q}, d_k:{self.C_k}, d_v:{self.C_v}')
        if self.C_q != self.C_k:
            raise ValueError('channels of "q" must equal channels of "k"')
        self.C_qk = self.C_q+dim_k[1]
        self.C_qkv = self.C_qk+dim_v[1]
        self.num_heads = num_heads
        self.scale = qk_scale or (self.C_q // num_heads) ** -0.5

        self.qkv = nn.Linear(dim_q[0], self.C_qkv, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_proj[0], dim_proj[1])
        self.proj_drop = nn.Dropout(proj_drop)

        self.n_token = n_token

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        qk = qkv[:, :, :self.C_qk].reshape(B, N, 2, self.num_heads, self.C_q // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = qkv[:, :, self.C_qk:].reshape(B, N, self.num_heads, self.C_v // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[-2], self.C_v)
        x = self.proj(x)

        x = self.proj_drop(x)

        return x, attn

class Mlp_cfged(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                act_layer=nn.GELU, 
                drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.out_features = out_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) if hidden_features else None
        self.act = act_layer() if hidden_features else None
        self.fc2 = nn.Linear(hidden_features, out_features) if out_features else None
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.out_features:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        return x
################################################################################################################
class Block_cfged(nn.Module):

    def __init__(self, dim_blk, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 n_tokens=196, cutoff=0.15): 
        super().__init__()
        dim_msa = [dim_blk['q'], dim_blk['k'], dim_blk['v'], dim_blk['proj']]
        self.n_tokens = n_tokens
        self.cutoff = cutoff

        self.norm1 = norm_layer(dim_blk['q'][0])
        self.attn = Attention_cfged(
            num_heads=dim_blk['h'][0], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            dim_msa=dim_msa, n_token=n_tokens
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim_blk['fc1'][0])
        self.mlp = Mlp_cfged(in_features=dim_blk['fc1'][0], hidden_features=dim_blk['fc1'][1], out_features=dim_blk['fc2'][1], act_layer=act_layer, drop=drop)

        self.gamma_1 = nn.Parameter(torch.ones((dim_blk['proj'][-1])),requires_grad=False)
        self.gamma_2 = nn.Parameter(torch.ones((dim_blk['fc2'][-1])),requires_grad=False)

    def forward(self, x):
        x_msa, attn = self.attn(self.norm1(x))
        x = self.drop_path(self.gamma_1 * x_msa) + x
        x = selector(x, attn, self.n_tokens, cutoff=self.cutoff)
        x = self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + x

        return x

################################################################################################################
class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, 
                #  =============================================================================================
                 dim_cfg=None, token_cfg=None, cutoff=0.15, output_mid=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_cfg = dim_cfg or self.build_cfg(in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, mlp_ratio=mlp_ratio)
        self.dim_cfg['head'] = (self.dim_cfg['head'][0], num_classes)
        self.patch_size = patch_size
        self.output_mid = output_mid
        self.depth = depth
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=self.dim_cfg['embed'][0], embed_dim=self.dim_cfg['embed'][1])
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=self.dim_cfg['embed'][0], embed_dim=self.dim_cfg['embed'][1])
        num_patches = self.patch_embed.num_patches

        self.num_tokens = token_cfg or [num_patches] * depth
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_cfg['embed'][1]))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.dim_cfg['embed'][1]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.dim = self.get_dim(depth=depth)
        self.blocks = nn.ModuleList([
            Block_cfged(
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                dim_blk=self.dim[i], n_tokens=self.num_tokens[i], cutoff=cutoff
                )
            for i in range(depth)])
        self.norm = norm_layer(self.dim_cfg['head'][0])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(self.dim_cfg['head'][0], self.dim_cfg['head'][1]) if num_classes > 0 else nn.Identity()

        # low frequency filtering
        self.output_midfeature = False
        self.midfeature_id = None

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def build_cfg(self, in_chans=3, num_classes=1000, embed_dim=768, depth=12, mlp_ratio=4.):
        dim_cfg = {'embed': (in_chans, embed_dim)}
        for i in range(depth):
            dim_cfg['q.'+str(i)] = (embed_dim, embed_dim)
            dim_cfg['k.'+str(i)] = (embed_dim, embed_dim)
            dim_cfg['v.'+str(i)] = (embed_dim, embed_dim)
            dim_cfg['h.'+str(i)] = (self.num_heads, )
            dim_cfg['proj.'+str(i)] = (embed_dim, embed_dim)
            dim_cfg['fc1.'+str(i)] = (embed_dim, round(embed_dim*mlp_ratio))
            dim_cfg['fc2.'+str(i)] = (round(embed_dim*mlp_ratio), embed_dim)
        dim_cfg['head'] = (embed_dim, num_classes)
        return dim_cfg
    
    def get_dim(self, depth):
        dim = []
        for i in range(depth):
            dim_dict = {'q': self.dim_cfg['q.'+str(i)]}
            dim_dict['k'] = self.dim_cfg['k.'+str(i)]
            dim_dict['v'] = self.dim_cfg['v.'+str(i)]
            dim_dict['h'] = self.dim_cfg['h.'+str(i)]
            dim_dict['proj'] = self.dim_cfg['proj.'+str(i)]
            dim_dict['fc1'] = self.dim_cfg['fc1.'+str(i)]
            dim_dict['fc2'] = self.dim_cfg['fc2.'+str(i)]
            dim.append(dim_dict)
        return dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        mid_feature = torch.zeros(1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.output_midfeature and i in self.midfeature_id:
                mid_feature = x[:, 0] if mid_feature.dim() == 1 else torch.cat((mid_feature, x[:, 0]), dim=0)

        x = self.norm(x)
        return x[:, 0], mid_feature

    def forward(self, x):
        x, x_mid = self.forward_features(x)
        if self.midfeature_id is not None and self.depth in self.midfeature_id:
           x_mid = x if x_mid.dim() == 1 else torch.cat((x_mid, x), dim=0)
        x = self.head(x)
        if self.output_mid:
            return x, x_mid
        else:
            return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def deit_tiny_cfged_patch16_224(pretrained=False, **kwargs):
    model = ViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_small_cfged_patch16_224(pretrained=False, **kwargs):
    model = ViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_base_cfged_patch16_224(pretrained=False, **kwargs):
    model = ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

