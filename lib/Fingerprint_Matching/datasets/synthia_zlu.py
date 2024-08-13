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

class Sythia_RidgeBase(Dataset):
    def __init__(self, ccr, split = "train"):
        if ccr: 
            self.base_path = "/panasas/scratch/grp-doermann/bhavin/FingerPrintData/"
        else:
            self.base_path = "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/"
        
        fingerdict = {
            "Index": 0,
            "Middle":1,
            "Ring": 2,
            "Little": 3
        }
        
        synthia_df = pd.read_csv("/home/spandey8/ridgebase_ICPR/Synthia_files_train.csv")

        self.split = split
        self.transforms ={
            "train": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    RandomRegionBlackOut(p=0.4, blackout_ratio=0.3),
                    RandomRegionBlurOut(p=0.4, blackout_ratio=0.3),
                    ## transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )]),
            "test": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224,224)),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                    ])
        }
        
        self.train_path = {
            "contactless": self.base_path + "Task1/Train/Contactless",
            "contactbased": self.base_path + "Task1/Train/Contactbased"
        }
        
        self.test_path = {
            "contactless": self.base_path + "Task1/Test/Contactless",
            "contactbased": self.base_path + "Task1/Test/Contactbased"
        }
        
        self.train_files = {
            "contactless": [self.train_path["contactless"] + "/" + f for f in os.listdir(self.train_path["contactless"]) if f.endswith('.png')],            
            "contactbased": [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.train_path["contactbased"]) for f in filenames if os.path.splitext(f)[1] == '.bmp']        
        }
        
        self.test_files = {
            "contactless": [self.test_path["contactless"] + "/" + f for f in os.listdir(self.test_path["contactless"]) if f.endswith('.png')],            
            "contactbased": [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.test_path["contactbased"]) for f in filenames if os.path.splitext(f)[1] == '.bmp']     
        }

        self.synthia_files = random.sample([(synthia_df["filename"][i], synthia_df["ids"][i]) for i in range(len(synthia_df))], 11252)
        self.synthia_base  = "/data2/scratch/Bhavin/Synthetic_Fingerprints/dataset/dataset"
        
        self.transform = self.transforms[split]
        self.allfiles = self.train_files if split == "train" else self.test_files
                
        self.label_id_mapping = set()
        
        self.label_id_to_contactbased = {}
        
        self.all_files_paths_contactless = []
        self.all_labels = []
        
        for filename in self.allfiles["contactless"]:
            id = filename.split("/")[-1].split("_")[2] + filename.split("/")[-1].split("_")[4].lower() + filename.split("/")[-1].split("_")[-1].split(".")[0]
            self.label_id_mapping.add(id)
            
        self.label_id_mapping = list(self.label_id_mapping)
        
        print(len(self.label_id_mapping))
        
        for filename in self.allfiles["contactless"]:
            id = filename.split("/")[-1].split("_")[2] + filename.split("/")[-1].split("_")[4].lower() + filename.split("/")[-1].split("_")[-1].split(".")[0]
            self.all_labels.append(self.label_id_mapping.index(id))
            self.all_files_paths_contactless.append(filename)
            
        for filename in self.allfiles["contactbased"]:
            id = filename.split("/")[-1].split("_")[1] + filename.split("/")[-1].split("_")[2].lower() + str(fingerdict[filename.split("/")[-1].split("_")[3].split(".")[0]])
            id = self.label_id_mapping.index(id)
            if (id in self.label_id_to_contactbased):
                self.label_id_to_contactbased[id].append(filename)
            else:
                self.label_id_to_contactbased[id] = [filename]
        
        print("Number of classes: ", len(self.label_id_mapping))
        print("Total number of images ", split ," : ", len(self.all_labels))
        
    def __len__(self):
        if (self.split == "train"):        
            print("Total Size: ", len(self.synthia_files), self.split)
            return len(self.synthia_files)
        else:
            print("Total Size: ", len(self.all_files_paths_contactless), self.split)
            return len(self.all_files_paths_contactless)
        
    def __getitem__(self, idx):
        if (self.split == "train"):
            map_idx_ridgebase = int(idx / (len(self.synthia_files) / len(self.all_files_paths_contactless)))
            
            if torch.is_tensor(map_idx_ridgebase):
                map_idx_ridgebase = map_idx_ridgebase.tolist()
            if torch.is_tensor(idx):
                idx = idx.tolist()
                
            label = self.all_labels[map_idx_ridgebase]
            contactless_filename  = self.all_files_paths_contactless[map_idx_ridgebase]
            contactbased_filename = self.label_id_to_contactbased[label][map_idx_ridgebase % len(self.label_id_to_contactbased[label])]
            
            contactless_sample  = cv2.imread(contactless_filename)
            contactbased_sample = cv2.imread(contactbased_filename)
            
            hand  = contactless_filename.split("/")[-1].split("_")[4]
            if hand == "RIGHT":
                contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_CLOCKWISE)
                contactless_sample = cv2.flip(contactless_sample, 1)
            else:
                contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_COUNTERCLOCKWISE)
                contactless_sample = cv2.flip(contactless_sample, 1)
        
            filename, syn_label = self.synthia_files[idx]
            synthia_sample  = cv2.imread(self.synthia_base + "/" + filename)
            # print(self.synthia_base + "/" + filename)
            synthia_label   = int(syn_label)
            
            if self.transform:
                contactless_sample  = self.transform(contactless_sample)
                contactbased_sample = self.transform(contactbased_sample)
                try:
                    synthia_sample      = self.transform(synthia_sample)
                except:
                    print(self.synthia_base + "/" + filename)
                
            return contactless_sample, contactbased_sample, self.all_labels[map_idx_ridgebase], synthia_sample, synthia_label + 10000 # 100000 is added to avoid intersection with Ridgebase label ids

        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            label               = self.all_labels[idx]
            contactless_filename  = self.all_files_paths_contactless[idx]
            contactbased_filename = self.label_id_to_contactbased[label][idx % len(self.label_id_to_contactbased[label])]
            
            contactless_sample  = cv2.imread(contactless_filename)
            contactbased_sample = cv2.imread(contactbased_filename)

            hand  = contactless_filename.split("/")[-1].split("_")[4]
            if hand == "RIGHT":
                contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_CLOCKWISE)
                contactless_sample = cv2.flip(contactless_sample, 1)
            else:
                contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_COUNTERCLOCKWISE)
                contactless_sample = cv2.flip(contactless_sample, 1)

            if self.transform:
                contactless_sample  = self.transform(contactless_sample)
                contactbased_sample = self.transform(contactbased_sample)

            return contactless_sample, contactbased_sample, self.all_labels[idx], contactless_filename, contactbased_filename
    
if __name__ == "__main__":
    ridgebase = RidgeBase_Pair(False, split = "train")
    dataloader = DataLoader(ridgebase, batch_size=4,
                        shuffle=True, num_workers=1)
    
    for image, label in dataloader:
        print(image.shape, label)