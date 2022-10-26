# from asyncio import current_task
# from mailbox import mbox
# from tkinter.messagebox import NO
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial import distance
import utils
from timm.models.vision_transformer import PatchEmbed, Attention
from torch import optim as optim
import itertools
import math
import json
from pathlib import Path
import copy
from torch.utils.data.sampler import Sampler

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
# ======================================================================================================================
def count_channels(model: nn.Module):
    total_channels = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            total_channels += m.weight.size(0)
    return int(total_channels)

def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)

def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def filtering(images, L=0.065, padding=0, reverse=False):
    if padding > 0:
        images = F.pad(images, pad=(padding, padding, padding, padding), mode='constant', value=0)

    _, _, H, W = images.shape
    K = min(H, W)
    d0 = (K * L / 2) ** 2
    m0 = (K - 1) / 2.
    x_coord = torch.arange(K).to(images.device)
    x_grid = x_coord.repeat(K).view(K, K)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    kernel = torch.exp(-torch.sum((xy_grid - m0)**2, dim=-1) / (2*d0))

    if IS_HIGH_VERSION:
        fftmaps = torch.fft.fft2(images)
        fftmaps = torch.stack([fftmaps.real, fftmaps.imag], -1)
        pha = torch.atan2(fftmaps[:,:,:,1], fftmaps[:,:,:,0])
        amp = torch.sqrt(fftmaps[:,:,:,0]**2 + fftmaps[:,:,:,1]**2)
    else:
        fftmaps = torch.rfft(images, signal_ndim=2, onesided=False)
        amp, pha = extract_ampl_phase(fftmaps)
    if reverse:
        mask = kernel
    else:
        mask = ifftshift(kernel)
    low_amp = amp.mul(mask)
    a1 = torch.cos(pha) * low_amp
    a2 = torch.sin(pha) * low_amp
    fft_src_ = torch.cat([a1.unsqueeze(-1),a2.unsqueeze(-1)],dim=-1)
    if IS_HIGH_VERSION:
        fft_src_ = torch.complex(fft_src_[..., 0], fft_src_[..., 1])
        outputs = torch.fft.ifft2(fft_src_)
    else:
        outputs = torch.irfft(fft_src_, signal_ndim=2, onesided=False)

    if padding > 0:
        outputs = outputs[:, :, padding:-padding, padding:-padding]

    return outputs

class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.as_tensor(range(len(self.indices))))

    def __len__(self):
        return len(self.indices)

class Pruning():
    def __init__(
        self, model: nn.Module, num_heads: int, num_classes: int, 
        prune_rate: float or list, prune_part: str, prune_block_id: list):
        super(Pruning, self).__init__()
        self.org_model = model
        self.n_heads = num_heads
        self.n_classes = num_classes
        self.prune_rate = prune_rate
        self.prune_part = prune_part.split(',')
        self.prune_block_id = [int(i) for i in prune_block_id.split(',')]
        self.score = None
        self.w_mgrad = None
        self.cor_mtx = []
        self.i_mask, self.o_mask, self.dim_cfg, self.token_cfg = [], [], {}, None
        self.head_cfg, self.head_mask = None, None
        self.actural_cpr = []
        self.n_samplers = 1000
        self.data_loader = None

        self.tau, self.alpha = 1.0, 0.1
        self.cutoff = 0.1
    
    def get_weight_mgrad(self, data_loader, device):
        teacher_model = copy.deepcopy(self.org_model)
        teacher_model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.org_model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
        self.org_model.train()
        grad_batch = {}
        dill_loss = 0
        for batch_id, (images, target) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                _, teacher_mid = teacher_model(images)

            self.org_model.train()
            optimizer.zero_grad()
            images = filtering(images, L=self.cutoff, padding=0, reverse=False)
            output, student_mid = self.org_model(images)
            loss = criterion(output, target)

            if student_mid.dim() > 1:
                T = self.tau
                distillation_loss = F.kl_div(
                    F.log_softmax(student_mid / T, dim=1),
                    F.log_softmax(teacher_mid / T, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (T * T) / student_mid.numel()
                dill_loss += distillation_loss.item()
                loss = loss * self.alpha + distillation_loss * (1 - self.alpha)

            loss.backward()

            if batch_id % 10 == 0:
                print(f'batch-{batch_id}: Counting weight grads... | distillation_loss: {dill_loss/(batch_id+1)}')
            for k, param in self.org_model.named_parameters():
                if param.requires_grad:
                    if batch_id == 0:
                        grad_batch[k] = param.grad
                    else:
                        grad_batch[k] = grad_batch[k] + param.grad
                  
        for k, v in grad_batch.items():
            grad_batch[k] = v / (batch_id + 1)

        return grad_batch

    def taylor_score(self, grad, param, name, block_id=None):
        w = (param.mul(grad[name]).view(param.shape[0], -1)**2).sum(-1)
        if 'qkv' in name:
            v_dim = self.org_model.dim_cfg['v.'+str(block_id)][-1]
            w_qk, w_v = w[:-v_dim].reshape(2, -1), w[-v_dim:]
            w = torch.cat((w_qk[0]+w_qk[1], w_qk[0]+w_qk[1], w_v), dim=0)
        return w

    def criterion_l_taylor(self, data_loader, device):
        score = []
        block_id = 0
        ori_graph_paras = self.org_model.state_dict()

        grad_dict = self.get_weight_mgrad(data_loader, device)
        self.w_mgrad = grad_dict
        print('Gradient achieved!!!')

        for k, v in ori_graph_paras.items():
            if k.find('proj') > -1 and k.find('embed') > -1 and v.dim() == 4:
                if 'embed' in self.prune_part:
                    w = self.taylor_score(grad=grad_dict, param=v, name=k)
                else:
                    w = v.view(v.shape[0], -1)[:, 0].mul(0).add(10e9).squeeze()
                score.append(w)
            elif k.find('blocks') > -1:
                if k.find('qkv') > -1 and v.dim() == 2:
                    if 'qkv' in self.prune_part and block_id in self.prune_block_id:
                        w = self.taylor_score(grad=grad_dict, param=v, name=k, block_id=block_id)
                    else:
                        w = v[:, 0].mul(0).add(10e9).squeeze()
                    score.append(w)
                elif k.find('proj') > -1 and v.dim() == 2: 
                    if 'proj' in self.prune_part and block_id in self.prune_block_id:
                        w = self.taylor_score(grad=grad_dict, param=v, name=k)
                    else:
                        w = v[:, 0].mul(0).add(10e9).squeeze()
                    score.append(w)
                elif k.find('fc1') > -1 and v.dim() == 2: 
                    if 'fc1' in self.prune_part and block_id in self.prune_block_id:
                        w = self.taylor_score(grad=grad_dict, param=v, name=k)
                    else:
                        w = v[:, 0].mul(0).add(10e9).squeeze()
                    score.append(w)
                elif k.find('fc2') > -1 and v.dim() == 2: 
                    if 'fc2' in self.prune_part and block_id in self.prune_block_id:
                        w = self.taylor_score(grad=grad_dict, param=v, name=k)
                    else:
                        w = v[:, 0].mul(0).add(10e9).squeeze()
                    score.append(w)
                    block_id += 1
        return score

    def get_score(self,  criterion: str, dataset=None, device=None, optimizer=None, args=None):
        if criterion == 'lfs':
            if self.data_loader is None:
                if self.n_samplers == len(dataset):
                    sampler = SubsetSampler(list(range(0, len(dataset), 10)))
                else:
                    sampler = torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(dataset)), self.n_samplers))
                data_loader = torch.utils.data.DataLoader(
                    dataset, sampler=sampler,
                    batch_size=100, num_workers=10,
                    pin_memory=True, drop_last=False)
            else:
                data_loader = self.data_loader
            self.score = self.criterion_l_taylor(data_loader, device)
        else:
            raise ValueError('Unsupported criterion!')

    def get_mask_and_newcfg(self, criterion: str, 
                            dataset=None, device=None, optimizer=None,
                            args=None, scores=None, dim_cfg=None):
        assert isinstance(self.prune_rate, float) and 0 < self.prune_rate < 1, f'unsupported pruning-rate: {self.prune_rate}'
        if scores is None:
            self.get_score(criterion, dataset, device, optimizer, args)
        else:
            self.score = scores.copy()
        ori_graph_paras = self.org_model.state_dict()
        self.i_mask, self.o_mask, self.dim_cfg = [], [], {}
        self.actural_cpr = []
        self.head_cfg = []
        print('Scores achieved!!!')

        score = None
        for s in self.score:
            score = s if score is None else torch.cat((score, s), dim=0)
        sorted_score, _ = torch.sort(score)
        threshold = float(sorted_score[round(score.size(0) * self.prune_rate)])
        print(f'Threshold achieved!!! threshold:{threshold}, prune_rate:{self.prune_rate}')

        idx = 0
        block_id = 0
        for k, v in ori_graph_paras.items():
            if k.find('embed') > -1 and k.find('mask') == -1:
                if k.find('proj') > -1 and k.find('weight') > -1:
                    self.i_mask.append(torch.ones(3))
                    mask_o = self.score[idx].ge(threshold)
                    self.o_mask.append(mask_o)
                    self.dim_cfg['embed'] = (int(self.i_mask[idx].sum()), int(self.o_mask[idx].sum()))
                    self.actural_cpr.append(float((v.size(0)-self.o_mask[idx].sum())/v.size(0)))
                    idx += 1
            elif k.find('blocks.') > -1 and k.find('mask') == -1:
                if k.find('qkv') > -1 and v.dim() == 2:
                    self.i_mask.append(self.o_mask[idx-1])
                    if block_id not in self.prune_block_id:
                        h_best = self.org_model.dim_cfg['h.'+str(block_id)][0]
                        sv_qk = self.org_model.dim_cfg['q.'+str(block_id)][-1] // h_best
                        sv_v = self.org_model.dim_cfg['v.'+str(block_id)][-1] // h_best
                        mask_o = torch.ones(v.shape[0])
                    else:
                        n_heads = self.org_model.dim_cfg['h.'+str(block_id)][0]
                        score = self.score[idx]
                        m_qkv = score.ge(threshold)
                        if self.org_model.dim_cfg['q.'+str(block_id)][-1] != self.org_model.dim_cfg['v.'+str(block_id)][-1]:
                            d_v = self.org_model.dim_cfg['v.'+str(block_id)][-1]
                            m_qk, m_v = m_qkv[:-d_v].reshape(int(n_heads*2), -1), m_qkv[-d_v:].reshape(n_heads, -1)
                            c_qk, c_v = m_qk.sum(-1), m_v.sum(-1)
                            c_q, c_k = c_qk[:n_heads],c_qk[n_heads:]
                            c_qkv = torch.cat((c_qk, c_v), dim=0)
                            s_qk, s_v = score[:-d_v].reshape(int(n_heads*2), -1), score[-d_v:].reshape(n_heads, -1)
                            s_q, s_k = s_qk[:n_heads], s_qk[n_heads:]
                            s_qkv = score
                            if c_v.sum() == 0:
                                _, id_v = torch.topk(s_v.view(-1), 1, dim=0, largest=True, sorted=False)
                                m_v = m_v.mul(0)
                                m_v[id_v] = 1
                                m_v = m_v.reshape(n_heads, -1)
                            if c_qk.sum() == 0:
                                _, id_qk = torch.topk(s_qk.view(-1), 2, dim=0, largest=True, sorted=False)
                                m_qk = m_qk.mul(0)
                                m_qk[id_qk] = 1
                                m_qk = m_qk.reshape(int(n_heads*2), -1)
                            c_qk, c_v = m_qk.sum(-1), m_v.sum(-1)
                            c_qkv = torch.cat((c_qk, c_v), dim=0)
                            m_qkv = torch.cat((m_qk.view(-1), m_v.view(-1)), dim=0)
                        else:
                            dim = int(v.size(0) // 3)
                            h_dim = dim // n_heads

                            m_qkv = m_qkv.reshape(-1, h_dim)
                            c_qkv = m_qkv.sum(-1)
                            m_qk, m_v = m_qkv[:n_heads*2], m_qkv[n_heads*2:]
                            c_qk, c_v = c_qkv[:n_heads*2], c_qkv[n_heads*2:]
                            c_q, c_k = c_qk[:n_heads],c_qk[n_heads:]

                            s_qkv = score.reshape(-1, h_dim)
                            s_qk, s_v = s_qkv[:n_heads*2], s_qkv[n_heads*2:]
                            s_q, s_k = s_qk[:n_heads], s_qk[n_heads:]

                        h_best = n_heads
                        print('Preparing to search number of heads...')
                        m_q, m_k = m_qk[:n_heads], m_qk[n_heads:]
                        self.head_mask = []
                        obj_score = float(m_qkv.mul(s_qkv).sum())
                        h_min = math.ceil(float(max(c_q.sum()/s_q.shape[-1], c_k.sum()/s_k.shape[-1], c_v.sum()/s_v.shape[-1])))
                        s_qk_sort, qk_id_sort = torch.sort(s_qk, dim=-1, descending=True)
                        s_v_sort, v_id_sort = torch.sort(s_v, dim=-1, descending=True)
                        qk_id_sort = qk_id_sort.reshape(2, n_heads, -1)
                        v_id_sort = v_id_sort.reshape(1, n_heads, -1)
                        loss_min = 1e9
                        m_h_best = torch.ones(n_heads)
                        h_idx = list(range(n_heads))
                        print(f'Starting searching number of heads, from {h_min}~{n_heads}...')
                        for h in range(h_min, n_heads+1):
                            M_idx = torch.as_tensor(list(itertools.combinations(h_idx, h))).to(v.device)
                            for i in range(M_idx.shape[0]):
                                M_tmp = torch.zeros(n_heads).to(v.device)
                                M_tmp[M_idx[i]] = 1
                                M = M_tmp[None,:] if i == 0 else torch.cat((M, M_tmp[None,:]), dim=0)

                            h_qk_dim, h_v_dim = max(round(float(c_qk.sum()/h/2)), 1), max(round(float(c_v.sum()/h)), 1)
                            s_qk_tmp, s_v_tmp = s_qk_sort.clone(), s_v_sort.clone()
                            s_qk_tmp, s_v_tmp = s_qk_tmp[:, :h_qk_dim].sum(-1), s_v_tmp[:, :h_v_dim].sum(-1)
                            s_qkv_cat = torch.cat((s_qk_tmp, s_v_tmp), dim=0) # 3n_h
                            s_qkv_cat = s_qkv_cat.reshape(3, -1) #3*n_h
                            tmp_score = (M[:, None].mul(s_qkv_cat[None])).sum(-1).sum(-1)#n_pl
                            tmp_score, t_id = torch.sort(tmp_score, descending=True)
                            loss_tmp = obj_score-float(tmp_score[0])

                            if loss_tmp < loss_min:
                                loss_min = loss_tmp
                                m_h_best = M[int(t_id[0])]
                                m_h_best = m_h_best[:, None].bool()
                                m_q.fill_(0), m_k.fill_(0), m_v.fill_(0)
                                q_id, k_id, v_id = qk_id_sort[0], qk_id_sort[1], v_id_sort[0]
                                q_id, k_id, v_id = q_id[:, :h_qk_dim], k_id[:, :h_qk_dim], v_id[:, :h_v_dim]
                                for i in range(n_heads):
                                    m_q[i][q_id[i]] = 1
                                    m_k[i][k_id[i]] = 1
                                    m_v[i][v_id[i]] = 1
                                m_q.mul_(m_h_best), m_k.mul_(m_h_best), m_v.mul_(m_h_best)
                                h_best = h
                        self.head_mask.append(m_h_best)
                        m_qk = torch.cat((m_q.view(-1), m_k.view(-1)), 0)
                        sv_qk, sv_v = m_qk.sum()/h_best/2, m_v.sum()/h_best
                        print('Best number of heads achieved...')

                        mask_o = torch.cat((m_qk.view(-1), m_v.view(-1)), 0)

                    self.o_mask.append(mask_o)
                    self.dim_cfg['q.'+str(block_id)] = (int(self.i_mask[idx].sum()), int(sv_qk*h_best))
                    self.dim_cfg['k.'+str(block_id)] = (int(self.i_mask[idx].sum()), int(sv_qk*h_best))
                    self.dim_cfg['v.'+str(block_id)] = (int(self.i_mask[idx].sum()), int(sv_v*h_best))
                    self.dim_cfg['h.'+str(block_id)] = (h_best, )
                    self.actural_cpr.append(float((v.size(0)-self.o_mask[idx].sum())/v.size(0)))
                    idx += 1
                elif k.find('proj') > -1 and v.dim() == 2:
                    dim = v.shape[1]
                    self.i_mask.append(self.o_mask[idx-1][-dim:])
                    mask_o = self.score[idx].ge(threshold)
                    self.o_mask.append(mask_o)
                    self.dim_cfg['proj.'+str(block_id)] = (int(self.i_mask[idx].sum()), int(self.o_mask[idx].sum()))
                    self.actural_cpr.append(float((v.size(0)-self.o_mask[idx].sum())/v.size(0)))
                    idx += 1
                elif k.find('fc1') > -1 and v.dim() == 2:
                    self.i_mask.append(self.o_mask[idx-1])
                    mask_o = self.score[idx].ge(threshold)
                    if mask_o.sum() == 0:
                        _, top_id = torch.topk(self.score[idx].view(-1), 1, dim=0, largest=True, sorted=False)
                        mask_o[top_id] = 1
                    self.o_mask.append(mask_o)
                    self.dim_cfg['fc1.'+str(block_id)] = (int(self.i_mask[idx].sum()), int(self.o_mask[idx].sum()))
                    self.actural_cpr.append(float((v.size(0)-self.o_mask[idx].sum())/v.size(0)))
                    idx += 1
                elif k.find('fc2') > -1 and v.dim() == 2:
                    self.i_mask.append(self.o_mask[idx-1])
                    mask_o = self.score[idx].ge(threshold)
                    self.o_mask.append(mask_o)
                    self.dim_cfg['fc2.'+str(block_id)] = (int(self.i_mask[idx].sum()), int(self.o_mask[idx].sum()))
                    self.actural_cpr.append(float((v.size(0)-self.o_mask[idx].sum())/v.size(0)))
                    idx += 1
                    block_id += 1
            elif k.find('head') > -1 and k.find('mask') == -1 and v.dim() == 2:
                self.i_mask.append(torch.ones(v.shape[1]))
                self.dim_cfg['head'] = (int(self.i_mask[-1].sum()), self.n_classes)

    def load_subgraph_from_model(self, sub_grah: nn.Module, pruned=False):
        ori_graph_paras = self.org_model.state_dict()
        sub_graph_paras = sub_grah.state_dict()
        layer_id = 0
        headm_id = 0

        if self.prune_rate > 0 or pruned:
        ############# load network parameters ##################
            for k, v in ori_graph_paras.items():
                if k.find('token') > -1 and k.find('mask') == -1:
                    if k in sub_graph_paras:
                        mask_o = self.o_mask[layer_id]
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        sub_graph_paras[k] = v[:,:, mask_o]

                elif k.find('embed') > -1 and k.find('mask') == -1:
                    if k in sub_graph_paras and k.find('weight') > -1:
                        mask_i, mask_o = self.i_mask[layer_id], self.o_mask[layer_id]
                        mask_i = mask_i.type('torch.BoolTensor').view(-1)
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        sub_graph_paras[k] = v[mask_o, :, :, :]
                        sub_graph_paras[k] = sub_graph_paras[k][:, mask_i, :, :]

                    elif k in sub_graph_paras and k.find('bias') > -1:
                        if v is not None:
                            mask_o = self.o_mask[layer_id]
                            mask_o = mask_o.type('torch.BoolTensor').view(-1)
                            sub_graph_paras[k] = v[mask_o]
                        layer_id += 1

                    elif k in sub_graph_paras:
                        mask_o = self.o_mask[layer_id]
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        sub_graph_paras[k] = v[:, :, mask_o]

                elif k.find('blocks') > -1 and k.find('mask') == -1:
                    if k in sub_graph_paras and k.find('norm') > -1:
                        if k.find('weight') > -1:
                            mask_o = self.i_mask[layer_id]
                            mask_o = mask_o.type('torch.BoolTensor').view(-1)
                            sub_graph_paras[k] = v[mask_o]
                        elif k.find('bias') > -1:
                            if v is not None:
                                mask_o = self.i_mask[layer_id]
                                mask_o = mask_o.type('torch.BoolTensor').view(-1)
                                sub_graph_paras[k] = v[mask_o]

                    elif k in sub_graph_paras and k.find('weight') > -1:
                        mask_i, mask_o = self.i_mask[layer_id], self.o_mask[layer_id]
                        mask_i = mask_i.type('torch.BoolTensor').view(-1)
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        sub_graph_paras[k] = v[mask_o, :]
                        sub_graph_paras[k] = sub_graph_paras[k][:, mask_i]

                    elif k in sub_graph_paras and k.find('bias') > -1:
                        if v is not None:
                            mask_o = self.o_mask[layer_id]
                            mask_o = mask_o.type('torch.BoolTensor').view(-1)
                            sub_graph_paras[k] = v[mask_o]
                        layer_id += 1

                    elif k in sub_graph_paras and k.find('head_mask') > -1:
                        sub_graph_paras[k] = v.mul(self.head_mask[headm_id])
                        headm_id += 1

                elif k.find('norm') > -1 and k.find('mask') == -1:
                    if k in sub_graph_paras:
                        if k.find('weight') > -1:
                            mask_o = self.i_mask[-1]
                            mask_o = mask_o.type('torch.BoolTensor').view(-1)
                            sub_graph_paras[k] = v[mask_o]
                        elif k.find('bias') > -1:
                            if v is not None:
                                mask_o = self.i_mask[-1]
                                mask_o = mask_o.type('torch.BoolTensor').view(-1)
                                sub_graph_paras[k] = v[mask_o]
                elif k.find('head') > -1 and k.find('mask') == -1:
                    if k in sub_graph_paras:
                        if k.find('weight') > -1:
                            mask_i = self.o_mask[-1]
                            mask_i = mask_o.type('torch.BoolTensor').view(-1)
                            sub_graph_paras[k] = v[:, mask_i]
                        elif k.find('bias') > -1:
                            sub_graph_paras[k] = v
                            
            sub_grah.load_state_dict(sub_graph_paras)
        
        else:
            sub_grah.load_state_dict(ori_graph_paras)

        return sub_grah

    def load_model_from_subgraph(self, sub_grah: nn.Module):
        ori_graph_paras = self.org_model.state_dict()
        sub_graph_paras = sub_grah.state_dict()
        layer_id = 0

        ############# load network parameters ##################
        for k, v in ori_graph_paras.items():
            if k.find('token') > -1 and k.find('mask') == -1:
                if k in sub_graph_paras:
                    mask_o = self.o_mask[layer_id]
                    mask_o = mask_o.type('torch.BoolTensor').view(-1)
                    v[:,:, mask_o] = sub_graph_paras[k]

            elif k.find('embed') > -1 and k.find('mask') == -1:
                if k in sub_graph_paras and k.find('weight') > -1:
                    # embed()
                    mask_i, mask_o = self.i_mask[layer_id], self.o_mask[layer_id]
                    mask_i = mask_i.type('torch.BoolTensor').view(-1)
                    mask_o = mask_o.type('torch.BoolTensor').view(-1)
                    middle_paras = v[mask_o, :, :, :]
                    middle_paras[:, mask_i, :, :] = sub_graph_paras[k]
                    v[mask_o, :, :, :] = middle_paras
                elif k in sub_graph_paras and k.find('bias') > -1:
                    if v is not None:
                        mask_o = self.o_mask[layer_id]
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        v[mask_o] = sub_graph_paras[k]
                    layer_id += 1
                elif k in sub_graph_paras:
                    mask_o = self.o_mask[layer_id]
                    mask_o = mask_o.type('torch.BoolTensor').view(-1)
                    v[:, :, mask_o] = sub_graph_paras[k]

            elif k.find('blocks') > -1 and k.find('mask') == -1:
                if k in sub_graph_paras and k.find('norm') > -1:
                    if k.find('weight') > -1:
                        mask_o = self.i_mask[layer_id]
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        v[mask_o] = sub_graph_paras[k]
                    elif k.find('bias') > -1:
                        if v is not None:
                            mask_o = self.i_mask[layer_id]
                            mask_o = mask_o.type('torch.BoolTensor').view(-1)
                            v[mask_o] = sub_graph_paras[k]
                elif k in sub_graph_paras and k.find('weight') > -1:
                    mask_i, mask_o = self.i_mask[layer_id], self.o_mask[layer_id]
                    mask_i = mask_i.type('torch.BoolTensor').view(-1)
                    mask_o = mask_o.type('torch.BoolTensor').view(-1)
                    middle_paras = v[mask_o, :] 
                    middle_paras[:, mask_i] = sub_graph_paras[k]
                    v[mask_o, :] = middle_paras
                elif k in sub_graph_paras and k.find('bias') > -1:
                    if v is not None:
                        mask_o = self.o_mask[layer_id]
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        v[mask_o] = sub_graph_paras[k]
                    layer_id += 1
            elif k.find('norm') > -1 and k.find('mask') == -1:
                if k in sub_graph_paras:
                    if k.find('weight') > -1:
                        mask_o = self.i_mask[-1]
                        mask_o = mask_o.type('torch.BoolTensor').view(-1)
                        v[mask_o] = sub_graph_paras[k]
                    elif k.find('bias') > -1:
                        if v is not None:
                            mask_o = self.i_mask[-1]
                            mask_o = mask_o.type('torch.BoolTensor').view(-1)
                            v[mask_o] = sub_graph_paras[k]
            elif k.find('head') > -1 and k.find('mask') == -1:
                if k in sub_graph_paras:
                    if k.find('weight') > -1:
                        mask_i = self.o_mask[-1]
                        mask_i = mask_o.type('torch.BoolTensor').view(-1)
                        v[:, mask_i] = sub_graph_paras[k]
                        
        self.org_model.load_state_dict(ori_graph_paras)
