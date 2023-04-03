import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import clip
from utils import *


def extract_few_shot_feature(cfg, clip_model, train_loader_cache):
    cache_keys = []
    cache_values = []
    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        
    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
    torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return


def extract_val_test_feature(cfg, split, clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
    torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    return


def extract_text_feature(cfg, classnames, prompt_path, clip_model, template):
    f = open(prompt_path)
    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts + cupl_texts
        
            texts_token = clip.tokenize(texts, truncate=True).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_cupl_t.pt")
    return


if __name__ == '__main__':
    
    clip_model, preprocess = clip.load('RN50')
    clip_model.eval()
    
    all_dataset = ["caltech101", 'dtd', 'eurosat', 'fgvc', 'food101', 'imagenet', 
                   'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']
    k_shot = [1, 2, 4, 8, 16]

    data_path = '/data0/zhuxiangyang/adapter/data'
    for set in all_dataset:
        
        cfg = yaml.load(open('configs/{}.yaml'.format(set), 'r'), Loader=yaml.Loader)

        cache_dir = os.path.join('./caches', cfg['dataset'])
        os.makedirs(cache_dir, exist_ok=True)
        cfg['cache_dir'] = cache_dir
        
        for k in k_shot:
            
            random.seed(1)
            torch.manual_seed(1)
            
            cfg['shots'] = k
            if set == 'imagenet':
                dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
                val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
                train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)           
            else:   
                dataset = build_dataset(set, data_path, k)
                val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

                train_tranform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
                train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)

            # Construct the cache model by few-shot training set
            print("\nConstructing cache model by few-shot visual features and labels.")
            extract_few_shot_feature(cfg, clip_model, train_loader_cache)
            
        # Extract val/test features
        print("\nLoading visual features and labels from val and test set.")
        extract_val_test_feature(cfg, "val", clip_model, val_loader)
        if not set == 'imagenet':
            extract_val_test_feature(cfg, "test", clip_model, test_loader)
                
        extract_text_feature(cfg, dataset.classnames, dataset.cupl_path, clip_model, dataset.template)
                
    
    