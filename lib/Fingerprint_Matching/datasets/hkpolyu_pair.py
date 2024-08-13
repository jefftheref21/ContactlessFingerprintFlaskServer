from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class RandomRegionBlackOut(object):
    def __init__(self, p=0.5, blackout_ratio=0.2):
        self.p = p  # Probability of applying the transform
        self.blackout_ratio = blackout_ratio  # Ratio of the image area to blackout

    def __call__(self, img):
        if random.random() < self.p:
            channels, width, height = img.shape
            mask_width              = int(width * self.blackout_ratio)
            mask_height             = int(height * self.blackout_ratio)

            start_x                 = random.randint(0, width - mask_width)
            start_y                 = random.randint(0, height - mask_height)

            img[:, start_x:start_x+mask_width, start_y:start_y+mask_height] = 0.0

        return img
    
class RandomRegionBlurOut(object):
    def __init__(self, p=0.5, blackout_ratio=0.2):
        self.p = p  # Probability of applying the transform
        self.blackout_ratio = blackout_ratio  # Ratio of the image area to blackout

    def __call__(self, img):
        if random.random() < self.p:
            channels, width, height = img.shape
            mask_width              = int(width * self.blackout_ratio)
            mask_height             = int(height * self.blackout_ratio)

            start_x                 = random.randint(0, width - mask_width)
            start_y                 = random.randint(0, height - mask_height)
            
            img[:, start_x:start_x+mask_width, start_y:start_y+mask_height] = transforms.GaussianBlur((3,3), sigma=(0.1, 2.0))(img[:, start_x:start_x+mask_width, start_y:start_y+mask_height])
        return img

class HKPolyU_Pair(Dataset):
    def __init__(self, ccr, split = "train"):
            self.split = split
            self.transforms ={
                "train": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                        RandomRegionBlurOut(p=0.4, blackout_ratio=0.2),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
                        transforms.RandomRotation((-30,30)),
                        transforms.Grayscale(num_output_channels=3),
                        ]),
                "val": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        transforms.Grayscale(num_output_channels=3),
                        ]),
                "test": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        transforms.Grayscale(num_output_channels=3),
                        ])
            }
            self.hkpoly_cb_train = pd.read_csv("/home/spandey8/ridgebase_ICPR/FM_bhavin/datasets/hkpoly_cb_train.csv")
            self.hkpoly_cb_val = pd.read_csv("/home/spandey8/ridgebase_ICPR/FM_bhavin/datasets/hkpoly_cb_val.csv")
            self.hkpoly_cb_test = pd.read_csv("/home/spandey8/ridgebase_ICPR/FM_bhavin/datasets/hkpoly_cb_test.csv")
            self.hkpoly_cl_train = pd.read_csv("/home/spandey8/ridgebase_ICPR/FM_bhavin/datasets/hkpoly_cl_train.csv")
            self.hkpoly_cl_val = pd.read_csv("/home/spandey8/ridgebase_ICPR/FM_bhavin/datasets/hkpoly_cl_val.csv")
            self.hkpoly_cl_test = pd.read_csv("/home/spandey8/ridgebase_ICPR/FM_bhavin/datasets/hkpoly_cl_test.csv")
            
            self.train_files = {
                "contactless": self.hkpoly_cl_train['Filename'].tolist(),
                "contactbased": self.hkpoly_cb_train['Filename'].tolist()
            }
            self.val_files = {
                "contactless": self.hkpoly_cl_val['Filename'].tolist(),
                "contactbased": self.hkpoly_cb_val['Filename'].tolist()
            }
            self.test_files = {
                "contactless": self.hkpoly_cl_test['Filename'].tolist(),
                "contactbased": self.hkpoly_cb_test['Filename'].tolist()
            }
            self.train_files_ids = {
                "contactless": self.hkpoly_cl_train['ids'].tolist(),
                "contactbased": self.hkpoly_cb_train['ids'].tolist()
            }
            self.val_files_ids = {
                "contactless": self.hkpoly_cl_val['ids'].tolist(),
                "contactbased": self.hkpoly_cb_val['ids'].tolist()
            }
            self.test_files_ids = {
                "contactless": self.hkpoly_cl_test['ids'].tolist(),
                "contactbased": self.hkpoly_cb_test['ids'].tolist()
            }
            
            self.transform = self.transforms[split]
            
            if self.split == 'train':
                self.allfiles = self.train_files
            elif self.split == 'val':
                self.allfiles = self.val_files
            else:
                self.allfiles = self.test_files
            
            if self.split == 'train':
                self.allids = self.train_files_ids
            elif self.split == 'val':
                self.allids = self.val_files_ids
            else:
                self.allids = self.test_files_ids
                
            self.label_id_to_contactbased = {}
            self.all_files_paths_contactless = self.allfiles['contactless']
            self.all_labels = self.allids['contactless']
            self.label_id_mapping = {val: ind for ind, val in enumerate(list(set(self.allids['contactless'])))}
            
            for cb_index in range(len(self.allids['contactbased'])):
                prev_id = self.allids['contactbased'][cb_index]
                id = self.label_id_mapping[prev_id]
                if id in self.label_id_to_contactbased:
                    self.label_id_to_contactbased[id].append(self.allfiles["contactbased"][cb_index])
                else:
                    self.label_id_to_contactbased[id] = [self.allfiles["contactbased"][cb_index]]
            
            self.all_labels = [self.label_id_mapping[i] for i in self.all_labels]
            print("Number of classes: ", len(self.label_id_mapping))
            print("Total number of images ", split ," : ", len(self.all_labels))
            
    def __len__(self):
        return len(self.all_files_paths_contactless)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.all_labels[idx]
        
        contactless_filename  = self.all_files_paths_contactless[idx]
        contactbased_filename = self.label_id_to_contactbased[label][idx % len(self.label_id_to_contactbased[label])]
        
        
        contactless_sample = cv2.imread(contactless_filename)
        contactbased_sample = cv2.imread(contactbased_filename)

        contactless_sample = cv2.flip(contactless_sample, 1)
        
        # print(contactless_filename, contactbased_filename, label)
        

        if self.transform:
            contactless_sample  = self.transform(contactless_sample)
            contactbased_sample = self.transform(contactbased_sample)

        return contactless_sample, contactbased_sample, label