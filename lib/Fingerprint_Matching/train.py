from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from datasets.ridgebase import RidgeBase
from datasets.ridgebase_pair import RidgeBase_Pair
from loss import DualMSLoss, DualMSLoss_FineGrained, SupConLoss, SupConLoss_MA, get_Arcface, get_MSloss, get_ProxyAnchor
import timm
from utils import Prev_RetMetric, RetMetric, compute_recall_at_k, l2_norm, compute_sharded_cosine_similarity, count_parameters
from pprint import pprint
import numpy as np
from tqdm import tqdm
from sampler import BalancedSampler
from torch.utils.data.sampler import BatchSampler
from torch.nn.parallel import DataParallel
from model import SwinModel as Model

def train(args, model, device, train_loader, optimizers, epoch, loss_func):
    model.train()
    for batch_idx, (x_cl, x_cb, target) in enumerate(pbar := tqdm(train_loader)):
        x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()
        x_cl, x_cb, x_cl_tokens, x_cb_tokens = model(x_cl, x_cb)
        loss = loss_func(x_cl, x_cb, x_cl_tokens, x_cb_tokens, target)
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

def test_finegrained(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_feats = []
    cl_feats = []
    cb_feats = []
    cl_labels = []
    cb_labels = []
    all_labels = []
    
    cl_tokens = []
    cb_tokens = []
    print("Computing Test Recall")
    with torch.no_grad():
        for (x_cl, x_cb, target) in tqdm(test_loader):
            x_cl, x_cb, target = x_cl.to(device), x_cb.to(device), target.to(device)
            x_cl, x_cb, x_cl_tokens, x_cb_tokens = model(x_cl, x_cb)
            x_cb = l2_norm(x_cb).cpu().detach()
            x_cl = l2_norm(x_cl).cpu().detach()
            target = target.cpu().detach()
            cl_feats.append(x_cl)
            cb_feats.append(x_cb)
            cl_labels.append(target)
            cb_labels.append(target)
            cl_tokens.append(x_cl_tokens.cpu().detach())
            cb_tokens.append(x_cb_tokens.cpu().detach())

    cl_feats = torch.cat(cl_feats)
    cb_feats = torch.cat(cb_feats)
    cl_labels = torch.cat(cl_labels)
    cb_labels = torch.cat(cb_labels)
    
    cl_tokens = torch.cat(cl_tokens)
    cb_tokens = torch.cat(cb_tokens)
    
    sim_mat = F.linear(l2_norm(cl_feats), l2_norm(cl_feats))
    sim_mat = sim_mat * (1 - torch.eye(sim_mat.shape[0], sim_mat.shape[1])) # remove diagonal for CL2CL
    cl2clk1 = compute_recall_at_k(sim_mat, cl_labels, cl_labels, 1)

    sim_mat = F.linear(l2_norm(cb_feats), l2_norm(cb_feats))
    sim_mat = sim_mat * (1 - torch.eye(sim_mat.shape[0], sim_mat.shape[1])) # remove diagonal for CL2CL
    cb2cbk1 = compute_recall_at_k(sim_mat, cb_labels, cb_labels, 1)
    
    sim_mat = F.linear(l2_norm(cl_feats), l2_norm(cb_feats)) # no need to remove diagonal for C2CL
    cl2cbk1 = compute_recall_at_k(sim_mat, cl_labels, cb_labels, 1)
    
    cl_tokens        = F.normalize(cl_tokens, dim=-1).cuda()
    cb_tokens        = F.normalize(cb_tokens, dim=-1).cuda()
    fin_sim_mat_clcb = compute_sharded_cosine_similarity(cl_tokens, cb_tokens, 20).cpu().detach()    
    clcbk1_token     = compute_recall_at_k(fin_sim_mat_clcb, cl_labels, cb_labels, 1)

    clcbk1_full      = compute_recall_at_k(fin_sim_mat_clcb + sim_mat, cl_labels, cb_labels, 1)

    print("Epoch: " + str(epoch) +  " Test CL2CL R@1: "      , cl2clk1 * 100,       "\n")
    print("Epoch: " + str(epoch) +  " Test CB2CB R@1: "      , cb2cbk1 * 100,       "\n")
    print("Epoch: " + str(epoch) +  " Test CL2CB R@1: "      , cl2cbk1 * 100,       "\n")
    print("Epoch: " + str(epoch) +  " Test CL2CB_Token R@1: ", clcbk1_token * 100,  "\n")
    print("Epoch: " + str(epoch) +  " Test CL2CB_Full R@1: " , clcbk1_full * 100,   "\n")
    torch.cuda.empty_cache()
    return cl2clk1, cl2cbk1, clcbk1_token

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
            x_cl, x_cb, _, _ = model(x_cl, x_cb)
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
    
    train_dataset    = RidgeBase_Pair(args.ccr, split="train")
    val_dataset      = RidgeBase_Pair(args.ccr, split="test")
    
    balanced_sampler = BalancedSampler(train_dataset, batch_size=args.batch_size, images_per_class = 3)
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
    
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    model = Model().to(device)
    ckpt = torch.load("/home/bhavinja/GradioDemoFingerprint/updated_demo/lib/Fingerprint_Matching/Models/ridgebase_train_vanilla_49_0.9111709286675639.pt", map_location=torch.device('cpu'))
    model.load_state_dict(ckpt, strict=False)

    print("Number of Trainable Parameters: - ", count_parameters(model))

    loss_func = DualMSLoss_FineGrained()

    optimizer_linear = optim.Adam(
        [
            {"params": model.linear_cl.parameters(), "lr":args.lr},
            # {"params": model.linear_cb.parameters(), "lr":args.lr},
         ],
        lr=args.lr)
    
    optimizer_swin = optim.Adam(
        [
            {"params": model.swin_cl.parameters(), "lr":args.lr * 0.01},
            {"params": model.swin_cb.parameters(), "lr":args.lr * 0.01}
         ],
        lr=args.lr)
        
    scheduler_linear = StepLR(optimizer_linear, step_size=5, gamma=args.gamma)
    scheduler_swin   = StepLR(optimizer_swin, step_size=5, gamma=args.gamma)

    cl2clk1, cl2cbk1, clcbk1_token = test_finegrained(model, device, test_loader, 0)

    for epoch in range(1, args.epochs + 1):            
        if (epoch > args.warmup):
            print("Training with Swin")
            train(args, model, device, train_loader, [optimizer_linear, optimizer_swin], epoch, loss_func)
        else:
            print("Training only linear")
            train(args, model, device, train_loader, [optimizer_linear], epoch, loss_func)
                
        if (epoch > args.warmup):
            scheduler_linear.step()
            scheduler_swin.step()
        else:
            scheduler_linear.step()
        
        if epoch % 1 == 0:    
            cl2clk1, cl2cbk1, clcbk1_token = test_finegrained(model, device, test_loader, epoch)
            recall1 = cl2clk1 + cl2cbk1 + clcbk1_token
        else:
            recall1 = 0
            
        torch.save(model.state_dict(), checkpoint_save_path + "Models/ridgebase_" + args.model_name + "_" + str(epoch) + "_" + str(recall1) + ".pt")


if __name__ == '__main__':
    main()