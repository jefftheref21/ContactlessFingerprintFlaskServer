from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# from datasets.ridgebase import RidgeBase
# from datasets.ridgebase_pair import RidgeBase_Pair
# from datasets.sythia_ridgeBase import Sythia_RidgeBase
from datasets.Syntha_hkpolyu import Syntha_HKPolyU
from loss import DualMSLoss, DualMSLoss_FineGrained, SupConLoss, SupConLoss_MA, get_Arcface, get_MSloss, get_ProxyAnchor, Domain_Loss, DualMSLoss_FineGrained_DA
import timm
from utils import Prev_RetMetric, RetMetric, compute_recall_at_k, l2_norm, compute_sharded_cosine_similarity, count_parameters
from pprint import pprint
import numpy as np
from tqdm import tqdm
from sampler import BalancedSampler
from torch.utils.data.sampler import BatchSampler
from torch.nn.parallel import DataParallel
from model import HKPolysynthModel_DA as Model   #check this
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json

def train(args, model, device, train_loader, optimizers, epoch, loss_func, domain_loss_func):
    model.train()
    for batch_idx, (x_cl, x_cb, target, s_sample, s_label) in enumerate(pbar := tqdm(train_loader)):
        
        x_cl, x_cb, target, s_sample, s_label = x_cl.to(device), x_cb.to(device), target.to(device), s_sample.to(device), s_label.to(device)
        
        for optimizer in optimizers:
            optimizer.zero_grad()
        # print(f"shape of contactless: {x_cl.shape}")
        # print(f"shape of contactbased: {x_cb.shape}")
        x_cl, x_cb, s_feat, x_cl_tokens, x_cb_tokens, s_sample_tokens, domain_scores = model(x_cl, x_cb, s_sample)
        
        contactbased_feat_batch = torch.cat([x_cb, s_feat])
        contactbased_token_feat_batch = torch.cat([x_cb_tokens, s_sample_tokens])
        contactbased_labels     = torch.cat([target, s_label])
        
        domain_scores = torch.cat(domain_scores)
        domain_target = torch.LongTensor([0] * x_cl.shape[0] + [1] * x_cb.shape[0] + [2] * s_sample.shape[0]).cuda()
        
        id_loss      = loss_func(x_cl, contactbased_feat_batch, x_cl_tokens, contactbased_token_feat_batch, target, contactbased_labels)
        domain_loss  = domain_loss_func(domain_scores, domain_target)
            
        loss = id_loss + domain_loss
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.dry_run:
                break
        pbar.set_description(f"Loss {loss}")

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_feats = []
    cl_feats = []
    cb_feats = []
    cl_labels = []
    cb_labels = []
    all_labels = []
    print("Computing Test Recall")
    with torch.no_grad():
        for (x_cl, x_cb, target) in tqdm(test_loader):
            x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
            x_cl, _ = model.get_embeddings(x_cl, type="contactless")
            x_cb, _ = model.get_embeddings(x_cb, type="contactbased")
            
            x_cb = l2_norm(x_cb).cpu().detach().numpy()
            x_cl = l2_norm(x_cl).cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            cl_feats.append(x_cl)
            cb_feats.append(x_cb)
            cl_labels.append(target)
            cb_labels.append(target)
            
    cl_feats = np.concatenate(cl_feats)
    cb_feats = np.concatenate(cb_feats)
    cl_labels = np.concatenate(cl_labels)
    cb_labels = np.concatenate(cb_labels)
    retmetric = Prev_RetMetric([cb_feats,cl_feats], [cb_labels,cl_labels], cl2cl=False)
    print("Epoch: " + str(epoch) +  " Test C2CL R@1: ", retmetric.recall_k(k=1) * 100, "\n")
    retmetric = Prev_RetMetric([cl_feats,cl_feats], [cl_labels,cl_labels], cl2cl=True)
    print("Epoch: " + str(epoch) +  " Test CL2CL R@1: ", retmetric.recall_k(k=1) * 100, "\n")
    return retmetric.recall_k(k=1)

def test_EER(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_feats = []
    cl_feats = []
    cb_feats = []
    cl_labels = []
    cb_labels = []
    all_labels = []
    cl_fnames = []
    cb_fnames = []
    print("Computing Test Recall")
    with torch.no_grad():
        for (x_cl, x_cb, target, cl_fname, cb_fname) in tqdm(test_loader):
            x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
            x_cl, _ = model.get_embeddings(x_cl, ftype="contactless")
            x_cb, _ = model.get_embeddings(x_cb, ftype="contactbased")
            x_cl = l2_norm(x_cl).cpu().detach().numpy()
            x_cb = l2_norm(x_cb).cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            cl_feats.append(x_cl)
            cb_feats.append(x_cb)
            cl_labels.append(target)
            cb_labels.append(target)
            cl_fnames.extend(cl_fname)
            cb_fnames.extend(cb_fname)
    cl_feats = torch.from_numpy(np.concatenate(cl_feats))
    cb_feats = torch.from_numpy(np.concatenate(cb_feats))
    cl_labels = torch.from_numpy(np.concatenate(cl_labels))
    cb_labels = torch.from_numpy(np.concatenate(cb_labels))

    with open("./misc/cl_fnames_best_hkpoly.txt", "w+") as file:
        for fname in cl_fnames:
            file.write(fname + "\n")

    unique_labels, indices  = torch.unique(cb_labels, return_inverse=True)
    unique_feats            = torch.stack([cb_feats[indices == i].mean(dim=0) for i in range(len(unique_labels))])
    cb_labels               = unique_labels
    cb_feats                = unique_feats
    cb_fnames_new           = [''] * len(cb_labels)
    # print(len(indices), min(indices), max(indices), len(cb_labels), len(cb_fnames))
    for index, i in enumerate(indices):
        cb_fnames_new[i.item()] = cb_fnames[index]
    with open("./misc/cb_fnames_hkpoly.txt", "w+") as file:
        for fname in cb_fnames_new:
            file.write(fname + "\n")
    
    scores = F.linear(cl_feats,cb_feats)
    np.save("./misc/task1_cb2cl_score_matrix_best_hkpoly.npy", scores)
    # print(scores.shape)
    scores = scores.flatten().tolist()
    ids = torch.eq(cl_labels.view(-1,1)-cb_labels.view(1,-1),0.0).flatten().tolist()
    # print(len(ids))
    ids_mod = list()
    for x in ids:
        if x==True:
            ids_mod.append(1)
        else:
            ids_mod.append(0)
    
    fpr, tpr, thresholds = roc_curve(ids_mod, scores)
    # with open("./evaluate/cl2cb_fpr_tpr_values_hkpoly.json", "w+") as file:
    #     json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, file)
    
    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
    tar_far_102 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve cl2cb task1 HK PolyU')
    plt.legend(loc="lower right")
    plt.savefig("./evaluate/roc_curve_cl2cb_task1_best_hkpoly.png", dpi=300, bbox_inches='tight')
    print(f"C2CL : EER: {EER * 100}")
    print(f"C2CL : TAR@FAR=10^-2 = ", tar_far_102 * 100)
    
    ################################################################################
    
    scores = F.linear(cl_feats,cl_feats)
    np.save("./misc/task1_cl2cl_score_matrix_best_hkpoly.npy", scores)
    scores = scores[torch.triu(torch.ones(scores.shape), diagonal=1) == 1].flatten().tolist()
    ids = torch.eq(cl_labels.view(-1,1)-cl_labels.view(1,-1),0.0).float().cuda()
    ids = ids[torch.triu(torch.ones(ids.shape), diagonal=1) == 1]
    ids_mod = list()
    for x in ids:
        if x==True:
            ids_mod.append(1)
        else:
            ids_mod.append(0)
    
    fpr, tpr, thresholds = roc_curve(ids_mod, scores)
    with open("./evaluate/cl2cl_fpr_tpr_values_best_hkpoly.json", "w+") as file:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, file)
    
    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
    tar_far_102 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve cl2cl task1 HK PolyU')
    plt.legend(loc="lower right")
    plt.savefig("./evaluate/roc_curve_cl2cl_task1_best_hkpoly.png", dpi=300, bbox_inches='tight')
    print(f"CL2CL : EER: {EER * 100}")
    print(f"CL2CL : TAR@FAR=10^-2 = ", tar_far_102 * 100)
    cl_labels = cl_labels.cpu().detach().numpy()
    cb_labels = cb_labels.cpu().detach().numpy()
    cl_feats  = cl_feats.cpu().detach().numpy()
    cb_feats  = cb_feats.cpu().detach().numpy()
    retmetric = Prev_RetMetric([cl_feats,cb_feats], [cl_labels,cb_labels], cl2cl=False)
    print("Epoch: " + str(epoch) +  " Test C2CL R@1: ", retmetric.recall_k(k=1) * 100, "\n")
    print("Epoch: " + str(epoch) +  " Test C2CL R@10: ", retmetric.recall_k(k=10) * 100, "\n")
    print("Epoch: " + str(epoch) +  " Test C2CL R@50: ", retmetric.recall_k(k=50) * 100, "\n")
    print("Epoch: " + str(epoch) +  " Test C2CL R@100: ", retmetric.recall_k(k=100) * 100, "\n")
    retmetric = Prev_RetMetric([cl_feats,cl_feats], [cl_labels,cl_labels], cl2cl=True)
    print("Epoch: " + str(epoch) +  " Test CL2CL R@1: ", retmetric.recall_k(k=1) * 100, "\n")
    print("Epoch: " + str(epoch) +  " Test CL2CL R@10: ", retmetric.recall_k(k=10) * 100, "\n")
    print("Epoch: " + str(epoch) +  " Test CL2CL R@50: ", retmetric.recall_k(k=50) * 100, "\n")
    print("Epoch: " + str(epoch) +  " Test CL2CL R@100: ", retmetric.recall_k(k=100) * 100, "\n")
    return retmetric.recall_k(k=1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ccr', action='store_true', default=False,
                        help='enables training on CCR')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--warmup', type=int, default=1, metavar='N',
                        help='warm up rate for feature extractor')
    parser.add_argument('--model-name', type=str, default="swinmodel",
                        help='Name of the model for checkpointing')
    
    args = parser.parse_args()
            
    if (args.ccr):
        checkpoint_save_path = "/panasas/scratch/grp-doermann/bhavin/FingerPrintData/"
    else:
        checkpoint_save_path = "./"

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_dataset    = Syntha_HKPolyU(args.ccr, split="train")
    # val_dataset      = Syntha_HKPolyU(args.ccr, split="val")
    test_dataset     = Syntha_HKPolyU(args.ccr, split="test")
    
    balanced_sampler = BalancedSampler(train_dataset, batch_size = args.batch_size, images_per_class = 2)
    batch_sampler    = BatchSampler(balanced_sampler, batch_size = args.batch_size, drop_last = True)
    
    train_kwargs     = {'batch_sampler': batch_sampler}
    test_kwargs      = {'batch_size':    args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
                       'num_workers': 1,
                       'pin_memory': True
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    print(len(train_loader))
    # val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    model = Model().to(device)
    ckpt = torch.load("/home/bhavinja/GradioDemoFingerprint/updated_demo/lib/Fingerprint_Matching/Models/ridgebase_BS64LR001NP10_2024_DA_Dual_FG_10_0.9120681920143562.pt", map_location=torch.device('cpu'))
    model.load_state_dict(ckpt,strict=False)

    print("Number of Trainable Parameters: - ", count_parameters(model))

    # loss_func = DualMSLoss_FineGrained_DA() #DualMSLoss_FineGrained_DA()
    loss_func   = DualMSLoss_FineGrained_DA()
    domain_loss = Domain_Loss()
    
    optimizer_linear = optim.Adam(
        [
            {"params": model.linear_cl.parameters(), "lr":args.lr},
            {"params": model.linear_cb.parameters(), "lr":args.lr},
            {"params": model.domain_predictor.parameters(), "lr":args.lr},
         ],
        lr=args.lr)
    
    optimizer_swin = optim.Adam(
        [
            {"params": model.swin_cl.parameters(), "lr":args.lr * 0.1},
        ],
        lr=args.lr)
        
    scheduler_linear = StepLR(optimizer_linear, step_size=10, gamma=args.gamma)
    scheduler_swin   = StepLR(optimizer_swin,   step_size=10, gamma=args.gamma)

    cl2clk1 = test_EER(model, device, test_loader, 0)
    # exit()

    for epoch in range(1, args.epochs + 1):            
        if (epoch > args.warmup):
            print("Training with Swin")
            train(args, model, device, train_loader, [optimizer_linear, optimizer_swin], epoch, loss_func, domain_loss)
        else:
            print("Training only linear")
            train(args, model, device, train_loader, [optimizer_linear], epoch, loss_func, domain_loss)
                
        if (epoch > args.warmup):
            scheduler_linear.step()
            scheduler_swin.step()
        else:
            scheduler_linear.step()
        
        cl2clk1 = test_EER(model, device, test_loader, epoch)
        torch.save(model.state_dict(), checkpoint_save_path + "Models/hkpoly_" + args.model_name + "_" + str(epoch) + "_" + str(cl2clk1) + ".pt")


if __name__ == '__main__':
    main()