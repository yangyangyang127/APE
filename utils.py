import os
import clip
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import glob
import numpy as np
from tqdm import tqdm


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=False):
    
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    cache_feat = cache_keys.reshape(cate_num, cfg['shots'], feat_dim)
    
    save_path = 'caches/{}'.format(cfg['dataset'])
    save_file = '{}/criterion_{}_{}shot.pt'.format(save_path, cfg['backbone'].replace('/', ''), cfg['shots'])
    
    if os.path.exists(save_file):
        print('Loading criterion...')
        sim = torch.load(save_file)
    elif only_use_txt:
        print('Calculating criterion...')
        
        feats = text_feat.squeeze()
        print(feats.shape)
        
        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                if i != j:
                    sim_sum += feats[i, :] * feats[j, :]
                    count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)
    else:
        print('Calculating criterion...')
        
        feats = torch.cat([text_feat, cache_feat], dim=1)
        samp_num = feats.shape[1]
        
        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                for m in range(samp_num):
                    for n in range(samp_num):
                        if i != j:
                            sim_sum += feats[i, m, :] * feats[i, n, :]
                            count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)

    criterion = (-1) * cfg['w'][0] * sim + cfg['w'][1] * torch.var(clip_weights, dim=1)
    _, indices = torch.topk(criterion, k=cfg['feat_num'])
    return indices


def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_cupl_t.pt"
    clip_weights = torch.load(save_path)
    return clip_weights


def load_few_shot_feature(cfg):
    cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return cache_keys, cache_values


def loda_val_test_feature(cfg, split):
    features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
    labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def accuracy(shot_logits, cache_values, topk=(1,)):
    target = cache_values.topk(max(topk), 1, True, True)[1].squeeze()
    pred = shot_logits.topk(max(topk), 1, True, True)[1].squeeze()
    idx = (target != pred)
    return idx

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
    
class APE_Training(nn.Module):
    def __init__(self, cfg, clip_weights, clip_model, cache_keys):
        super(APE_Training, self).__init__()
        self.shots = cfg['shots']
        self.feat_dim, self.cate_num = clip_weights.shape
        
        self.value_weights = nn.Parameter(torch.ones([self.cate_num*cfg['shots'], 1]).half().cuda(), requires_grad=True)
        self.indices = cal_criterion(cfg, clip_weights, cache_keys)

        self.res = nn.Parameter(torch.zeros([self.cate_num, cfg['feat_num']]).half().cuda(), requires_grad=True)
        self.feat_num = cfg['feat_num']
        
    def forward(self, cache_keys, clip_weights, cache_values):
        
        res_keys = self.res.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.feat_dim)
        new_cache_keys[:, self.indices] = new_cache_keys[:, self.indices] + res_keys
    
        res_text = self.res.t()
        new_clip_weights = clip_weights.clone()
        new_clip_weights[self.indices, :] = clip_weights[self.indices, :] + res_text 
        new_cache_values = cache_values * self.value_weights
       
        return new_cache_keys.half(), new_clip_weights.half(), new_cache_values.half()
    
    
    
