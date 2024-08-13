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
from PIL import Image
import random
from torchvision.utils import save_image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def cropping_preprocess(image):
    non_zero_pixels = np.where(image != 255)
    y_min, x_min = np.min(non_zero_pixels[0]), np.min(non_zero_pixels[1])
    y_max, x_max = np.max(non_zero_pixels[0]), np.max(non_zero_pixels[1])
    top_left = (x_min, y_min)
    top_right = (x_max, y_min)
    bottom_left = (x_min, y_max)
    bottom_right = (x_max, y_max)
    height = bottom_right[1] - top_left[1] + 1
    width = bottom_right[0] - top_left[0] + 1
    cropped_img = image[top_left[1]:top_left[1] + height, top_left[0]:top_left[0] + width]

    h,w = cropped_img.shape[:2]
    if h>224 and w>224:
        return cropped_img
    else:
        scale_factor_h = 224 / h
        scale_factor_w = 224 / w
        new_width = int(w * scale_factor_w)
        new_height = int(h * scale_factor_h)
        resized_image = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # print(resized_image.shape)
        return resized_image

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

class RidgeBase_HKPolyU_Synthia(Dataset):
    def __init__(self, ccr, split = "train"):
            self.split = split

            fingerdict = {
                "Index": 0,
                "Middle":1,
                "Ring": 2,
                "Little": 3
            }

            self.transforms ={
                "train": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        RandomRegionBlackOut(p=0.4, blackout_ratio=0.4),
                        RandomRegionBlurOut(p=0.4, blackout_ratio=0.4),
                        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
                        transforms.Grayscale(num_output_channels=3),
                        ]),
                "val": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        transforms.Grayscale(num_output_channels=3)
                        ]),
                "test": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        transforms.Grayscale(num_output_channels=3)])
            }
            
            synthia_df = pd.read_csv("/home/spandey8/ridgebase_ICPR/Synthia_files_train.csv")

            self.hkpoly_cb_train = pd.read_csv("./datasets/hkpolyU/hkpoly_cb_train.csv")
            self.hkpoly_cb_val = pd.read_csv("./datasets/hkpolyU/hkpoly_cb_val.csv")
            self.hkpoly_cb_test = pd.read_csv("./datasets/hkpolyU/hkpoly_cb_test.csv")

            self.hkpoly_cl_train = pd.read_csv("./datasets/hkpolyU/hkpoly_cl_train.csv")
            self.hkpoly_cl_val = pd.read_csv("./datasets/hkpolyU/hkpoly_cl_val.csv")
            self.hkpoly_cl_test = pd.read_csv("./datasets/hkpolyU/hkpoly_cl_test.csv")

            self.ridgebase_cl_train = pd.read_csv("./datasets/ridgebase_dataset/ridgebase_cl_train.csv")
            self.ridgebase_cl_test  = pd.read_csv("./datasets/ridgebase_dataset/ridgebase_cl_test.csv")
            self.ridgebase_cb_train = pd.read_csv("./datasets/ridgebase_dataset/ridgebase_cb_train.csv")
            self.ridgebase_cb_test  = pd.read_csv("./datasets/ridgebase_dataset/ridgebase_cb_test.csv")

            self.train_files = {
                "contactless":  self.hkpoly_cl_train['Filename'].tolist() + self.ridgebase_cl_train['Filename'].tolist(),
                "contactbased": self.hkpoly_cb_train['Filename'].tolist() + self.ridgebase_cb_train['Filename'].tolist()
            }
            self.test_files = {
                "contactless":  self.hkpoly_cl_test['Filename'].tolist(), #+ self.ridgebase_cl_test['Filename'].tolist(),
                "contactbased": self.hkpoly_cb_test['Filename'].tolist() #+ self.ridgebase_cb_test['Filename'].tolist(),
            }

            # print([(fname.split("/")[-1].split(".")[0].split("_")[2] + fname.split("/")[-1].split(".")[0].split("_")[4] + fname.split("/")[-1].split(".")[0].split("_")[7]).lower() + "_rb" for fname in self.ridgebase_cl_train['Filename'].tolist()])
            self.train_files_ids = {
                "contactless":  [str(id)+"_hk" for id in self.hkpoly_cl_train['ids'].tolist()] + [(fname.split("/")[-1].split(".png")[0].split("_")[2] + fname.split("/")[-1].split(".png")[0].split("_")[4] + fname.split("/")[-1].split(".png")[0].split("_")[8]).lower() + "_rb" for fname in self.ridgebase_cl_train['Filename'].tolist()],
                "contactbased": [str(id)+"_hk" for id in self.hkpoly_cb_train['ids'].tolist()] + ["".join(fname.split("/")[-1].split(".")[0].split("_")[1:-1]).lower() + "" + str(fingerdict[fname.split("/")[-1].split(".")[0].split("_")[-1]]) +"_rb" for fname in self.ridgebase_cb_train['Filename'].tolist()]
            }
            self.test_files_ids = {
                "contactless":  [str(id)+"_hk" for id in self.hkpoly_cl_test['ids'].tolist()], #+ [(fname.split("/")[-1].split(".")[0].split("_")[2] + fname.split("/")[-1].split(".")[0].split("_")[4] + fname.split("/")[-1].split(".")[0].split("_")[7]).lower() + "_rb" for fname in self.ridgebase_cl_test['Filename'].tolist()],
                "contactbased": [str(id)+"_hk" for id in self.hkpoly_cb_test['ids'].tolist()] #+ ["".join(fname.split("/")[-1].split(".")[0].split("_")[1:-1]).lower() + "" + str(fingerdict[fname.split("/")[-1].split(".")[0].split("_")[-1]]) +"_rb" for fname in self.ridgebase_cb_test['Filename'].tolist()]
            }

            print(len(self.test_files["contactless"]), len(self.test_files_ids["contactless"]))
            print(len(self.train_files["contactless"]), len(self.train_files_ids["contactless"]))
            print(len(self.test_files["contactbased"]), len(self.test_files_ids["contactbased"]))
            print(len(self.train_files["contactbased"]), len(self.train_files_ids["contactbased"]))
            print(len(set(self.test_files_ids["contactless"])), len(set(self.test_files_ids["contactbased"])))
            print(len(set(self.train_files_ids["contactless"])), len(set(self.train_files_ids["contactbased"])))

            self.synthia_files = random.sample([(synthia_df["filename"][i], synthia_df["ids"][i]) for i in range(len(synthia_df))], 13172)
            self.synthia_base  = "/data2/scratch/Bhavin/Synthetic_Fingerprints/dataset/dataset"
            
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
                
            label                 = self.all_labels[map_idx_ridgebase]
            contactless_filename  = self.all_files_paths_contactless[map_idx_ridgebase]
            contactbased_filename = self.label_id_to_contactbased[label][map_idx_ridgebase % len(self.label_id_to_contactbased[label])]

            contactless_filename = contactless_filename.replace("hkpoly_data/cl", "hkpoly_data/cl_png/cl/")
            contactless_filename = contactless_filename.replace(".bmp", ".png")
            
            # contactless_sample  = cv2.imread(contactless_filename)
            contactless_sample  = cv2.imread(contactless_filename)
            contactbased_sample = cv2.imread(contactbased_filename)
            contactbased_sample = cropping_preprocess(contactbased_sample)

            if ("LEFT" in contactless_filename or "RIGHT" in contactless_filename):
                hand = contactless_filename.split("/")[-1].split("_")[4]
                if hand == "RIGHT":
                    contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_CLOCKWISE)
                    contactless_sample = cv2.flip(contactless_sample, 1)
                else:
                    contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    contactless_sample = cv2.flip(contactless_sample, 1)

            filename, syn_label = self.synthia_files[idx]
            synthia_sample  = cv2.imread(self.synthia_base + "/" + filename)
            synthia_sample  = cropping_preprocess(synthia_sample)
            synthia_label   = int(syn_label)
            
            if self.transform:
                contactless_sample      = self.transform(contactless_sample)
                contactbased_sample     = self.transform(contactbased_sample)
                try:
                    synthia_sample      = self.transform(synthia_sample)
                except:
                    print(self.synthia_base + "/" + filename)

    
            # print(contactbased_sample.shape, contactless_sample.shape)
            # print("________________________")
            # utils.save_image(utils.make_grid(torch.clip(contactless_sample.data, 0, 1)), "/home/spandey8/ridgebase_ICPR/FM_bhavin/train_images_screening/"+os.path.splitext(contactless_filename.split("/")[-1])[0]+"_contactless.jpg")
            # utils.save_image(utils.make_grid(torch.clip(contactbased_sample.data, 0, 1)), "/home/spandey8/ridgebase_ICPR/FM_bhavin/train_images_screening/"+os.path.splitext(contactbased_filename.split("/")[-1])[0]+"_contactbased.jpg")
            # exit(0)

            return contactless_sample, contactbased_sample, self.all_labels[map_idx_ridgebase], synthia_sample, synthia_label + 999999 # 100000 is added to avoid intersection with Ridgebase label ids

        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            label               = self.all_labels[idx]
            contactless_filename  = self.all_files_paths_contactless[idx]
            # print(contactless_filename)
            contactbased_filename = self.label_id_to_contactbased[label][idx % len(self.label_id_to_contactbased[label])]

            contactless_filename = contactless_filename.replace("hkpoly_data/cl", "hkpoly_data/cl_png/cl/")
            contactless_filename = contactless_filename.replace(".bmp", ".png")
            # print(contactless_filename)
            contactless_sample  = cv2.imread(contactless_filename)
            contactless_sample = contactless_sample
            # print(len(contactless_sample.getbands()))
            # print(contactless_filename)
            # contactless_sample.save("15_contactless.bmp")
            
            contactbased_sample = cv2.imread(contactbased_filename)
            contactbased_sample = cropping_preprocess(contactbased_sample)
            # print("__________________")
            # print(contactbased_sample.shape)
            if self.transform:
                contactless_sample  = self.transform(contactless_sample)
                contactbased_sample = self.transform(contactbased_sample)
            
            # save_image(contactbased_sample, "contactbased.jpg")
            # save_image(contactless_sample, "contactless.jpg")
            # exit(0)

            return contactless_sample, contactbased_sample, self.all_labels[idx], contactless_filename, contactbased_filename
    

if __name__ == "__main__":
    hkpoly = RidgeBase_HKPolyU_Synthia(False, split = "train")
    dataloader = DataLoader(hkpoly, batch_size=4,shuffle=True, num_workers=1)
    for x_cl, x_cb, target, s_sample, s_label in dataloader:
        print(x_cl.shape)
        print(x_cb.shape)
    # img = "/home/spandey8/ridgebase_ICPR/FM_bhavin/hkpoly_data/cb/train/172_11.jpg"
    # img = cv2.imread(img)
    # cropped_img = cropping_preprocess(img)
    # cv2.imwrite("cropped.jpg",cropped_img)
    # print(cropped_img.shape)