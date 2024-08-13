from pytorch_metric_learning import losses 
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import os
import torch.nn.functional as F
import itertools

torch.autograd.set_detect_anomaly(True)

def get_MSloss():
    msloss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    return msloss

def get_Arcface(num_classes, embedding_size):
    msloss = losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64)
    return msloss

def get_ProxyAnchor(num_classes, embedding_size):
    proxyanchor = losses.ProxyAnchorLoss(num_classes, embedding_size, margin = 0.1, alpha = 32)
    return proxyanchor

class SupConLoss(nn.Module):
    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output
    def __init__(self, margin=0, max_violation=False):
        super(SupConLoss, self).__init__()
        self.loss_func = losses.SupConLoss(temperature=0.3).to(torch.device('cuda'))
        # self.loss_func = losses.CrossBatchMemory(self.loss_func, 1024, memory_size=2500, miner=None)

    def forward(self, x_contactless, x_contactbased, x_cl_tokens, x_cb_tokens, labels, labels_contactbased):
        # x_contactbased = self.l2_norm(x_contactbased)
        # x_contactbased = self.l2_norm(x_contactbased)
        embeddings = torch.cat([x_contactless, x_contactbased], dim = 0)
        labels = torch.cat([labels, labels_contactbased])
        loss   = self.loss_func(embeddings, labels)
        return loss
    
class SupConLoss_MA(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(SupConLoss_MA, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(1024, 1, batch_first=True)
        self.linear = nn.Linear(1024, 1)
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.5
        self.margin = 0.1
        self.regions = 36
        self.scale_pos = 2.0
        self.scale_neg = 40.0
        self.scale_pos_p = 2.0
        self.scale_neg_p = 40.0
        
    def ms_sample(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        return pos_loss + neg_loss

    def get_all_pairs(self, input_tensor1, input_tensor2):
        B, N, d = input_tensor2.shape
        pairs = list(itertools.combinations(range(B), 2))
        output_tensor = torch.zeros(B, B, N, d, 2)

        for i, (idx1, idx2) in enumerate(pairs):
            output_tensor[idx1, idx2] = torch.stack([input_tensor1[idx1], input_tensor2[idx2]], dim=-1)
            output_tensor[idx2, idx1] = torch.stack([input_tensor1[idx2], input_tensor2[idx1]], dim=-1)

        return output_tensor
    
    def forward(self, x_contactless, x_contactbased, x_contactless_tokens, x_contactbased_tokens, labels):
        # print(x_contactless_tokens.shape)
        batch          = x_contactless.shape[0]
        pairs          = self.get_all_pairs(x_contactless_tokens, x_contactbased_tokens)
        output_tensor  = pairs.view(pairs.shape[0] * pairs.shape[1], pairs.shape[2] * pairs.shape[4], pairs.shape[3]).cuda()
        print("somewhere", output_tensor.shape)
        attn_output, _ = self.multihead_attn(output_tensor, output_tensor, output_tensor)
        print("somewhere")
        match_vector   = attn_output.mean(dim=1)
        match_score    = F.sigmoid(self.linear(match_vector)).squeeze(dim=1)
        match_score    = match_score.view(batch, batch)
        loss           = self.ms_sample(match_score, labels)
        return loss

class DualMSLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(DualMSLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.5
        self.margin = 0.1
        self.regions = 36
        self.scale_pos = 2.0
        self.scale_neg = 40.0
        self.scale_pos_p = 2.0
        self.scale_neg_p = 40.0

    def ms_sample(self,sim_mat, label1, label2):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label1.view(-1,1)-label2.view(1,-1),0.0).float().cuda()
        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def forward(self, x_contactless, x_contactbased, labels, labels_contactbased):        
        sim_mat = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactbased))
        loss1 = self.ms_sample(sim_mat, labels, labels_contactbased).cuda() + self.ms_sample(sim_mat.t(), labels_contactbased, labels).cuda()

        sim_mat = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactless))
        loss2 = self.ms_sample(sim_mat, labels, labels).cuda() + self.ms_sample(sim_mat.t(), labels, labels).cuda()

        sim_mat = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactbased))
        loss3 = self.ms_sample(sim_mat, labels_contactbased, labels_contactbased).cuda() + self.ms_sample(sim_mat.t(), labels_contactbased, labels_contactbased).cuda()

        return loss1 + loss3 + loss2

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output
    
class Domain_Loss(nn.Module):
    def __init__(self):
        super(Domain_Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, domain_pred, domains_target):
        return self.cross_entropy(domain_pred, domains_target)

class DualMSLoss_FineGrained_DA(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(DualMSLoss_FineGrained_DA, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.8
        self.margin = 0.3
        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def ms_sample(self,sim_mat,label,label2):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label2.view(1,-1),0.0).float().cuda()
        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def compute_sharded_cosine_similarity(self, tensor1, tensor2, shard_size):
        B, T, D = tensor1.shape
        # print(tensor1.shape, tensor2.shape)
        average_sim_matrix = torch.zeros((B, tensor2.shape[0]), device=tensor1.device)

        for start_idx in range(0, T, shard_size):
            end_idx = min(start_idx + shard_size, T)

            # Get the shard
            shard_tensor1 = tensor1[:, start_idx:end_idx, :]
            shard_tensor2 = tensor2[:, start_idx:end_idx, :]

            # Reshape and expand
            shard_tensor1_expanded = shard_tensor1.unsqueeze(1).unsqueeze(3)
            shard_tensor2_expanded = shard_tensor2.unsqueeze(0).unsqueeze(2)

            # Compute cosine similarity for the shard
            shard_cos_sim = F.cosine_similarity(shard_tensor1_expanded, shard_tensor2_expanded, dim=-1)

            # Accumulate the sum of cosine similarities
            average_sim_matrix += torch.sum(shard_cos_sim, dim=[2, 3])

        # Normalize by the total number of elements (T*T)
        average_sim_matrix /= (T * T)

        return average_sim_matrix

    def forward(self, x_contactless, x_contactbased, x_cl_tokens, x_cb_tokens, labels, labels_contactbased):
                
        sim_mat_clcl = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactless))
        sim_mat_cbcb = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactbased))
        sim_mat_clcb = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactbased))

        # x_cl_tokens        = F.normalize(x_cl_tokens, dim=-1)#.detach().cpu()
        # x_cb_tokens        = F.normalize(x_cb_tokens, dim=-1)#.detach().cpu()
        # fin_sim_mat_clcb   = self.compute_sharded_cosine_similarity(x_cl_tokens, x_cb_tokens, 10)

        # print(labels.shape, labels_contactbased.shape)
        
        loss1              = self.ms_sample(sim_mat_clcb, labels, labels_contactbased).cuda() + self.ms_sample(sim_mat_clcb.t(), labels_contactbased, labels).cuda()
        loss2              = self.ms_sample(sim_mat_clcl, labels, labels).cuda() + self.ms_sample(sim_mat_clcl.t(), labels, labels).cuda()
        loss3              = self.ms_sample(sim_mat_cbcb, labels_contactbased, labels_contactbased).cuda() + self.ms_sample(sim_mat_cbcb.t(), labels_contactbased, labels_contactbased).cuda()
        # loss4              = self.ms_sample(fin_sim_mat_clcb, labels, labels_contactbased).cuda() + self.ms_sample(fin_sim_mat_clcb.t(), labels_contactbased, labels).cuda()

        return loss1 + loss2 + loss3 #+ loss4

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output
    
class DualMSLoss_FineGrained(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(DualMSLoss_FineGrained, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.5
        self.margin = 0.1
        self.regions = 36
        self.scale_pos = 2.0
        self.scale_neg = 40.0
        self.scale_pos_p = 2.0
        self.scale_neg_p = 40.0

    def ms_sample(self,sim_mat,label):
        pos_exp = torch.exp(-self.scale_pos*(sim_mat-self.thresh))
        neg_exp = torch.exp( self.scale_neg*(sim_mat-self.thresh))
        pos_mask = torch.eq(label.view(-1,1)-label.view(1,-1),0.0).float().cuda()
        neg_mask = 1-pos_mask
        P_sim = torch.where(pos_mask == 1,sim_mat,torch.ones_like(pos_exp)*1e16)
        N_sim = torch.where(neg_mask == 1,sim_mat,torch.ones_like(neg_exp)*-1e16)
        min_P_sim,_ = torch.min(P_sim,dim=1,keepdim=True)
        max_N_sim,_ = torch.max(N_sim,dim=1,keepdim=True)
        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss = torch.log(1+hard_P_sim).sum()/self.scale_pos
        neg_loss = torch.log(1+hard_N_sim).sum()/self.scale_neg
        
        return pos_loss + neg_loss

    def compute_sharded_cosine_similarity(self, tensor1, tensor2, shard_size):
        B, T, D = tensor1.shape
        average_sim_matrix = torch.zeros((B, B), device=tensor1.device)

        for start_idx in range(0, T, shard_size):
            end_idx = min(start_idx + shard_size, T)

            # Get the shard
            shard_tensor1 = tensor1[:, start_idx:end_idx, :]
            shard_tensor2 = tensor2[:, start_idx:end_idx, :]

            # Reshape and expand
            shard_tensor1_expanded = shard_tensor1.unsqueeze(1).unsqueeze(3)
            shard_tensor2_expanded = shard_tensor2.unsqueeze(0).unsqueeze(2)

            # Compute cosine similarity for the shard
            shard_cos_sim = F.cosine_similarity(shard_tensor1_expanded, shard_tensor2_expanded, dim=-1)

            # Accumulate the sum of cosine similarities
            average_sim_matrix += torch.sum(shard_cos_sim, dim=[2, 3])

        # Normalize by the total number of elements (T*T)
        average_sim_matrix /= (T * T)

        return average_sim_matrix

    def forward(self, x_contactless, x_contactbased, x_cl_tokens, x_cb_tokens, labels):
                
        sim_mat_clcl = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactless))
        sim_mat_cbcb = F.linear(self.l2_norm(x_contactbased), self.l2_norm(x_contactbased))

        x_cl_tokens        = F.normalize(x_cl_tokens, dim=-1)#.detach().cpu()
        x_cb_tokens        = F.normalize(x_cb_tokens, dim=-1)#.detach().cpu()
        fin_sim_mat_clcb   = self.compute_sharded_cosine_similarity(x_cl_tokens, x_cb_tokens, 10)

        sim_mat_clcb       = F.linear(self.l2_norm(x_contactless), self.l2_norm(x_contactbased))
        clcb_weighted      = sim_mat_clcb + fin_sim_mat_clcb
        
        loss1              = self.ms_sample(clcb_weighted, labels).cuda() + self.ms_sample(clcb_weighted.t(), labels).cuda()
        # loss2              = self.ms_sample(sim_mat_clcl, labels).cuda() + self.ms_sample(sim_mat_clcl.t(), labels).cuda()
        # loss3              = self.ms_sample(sim_mat_cbcb, labels).cuda() + self.ms_sample(sim_mat_cbcb.t(), labels).cuda()

        return loss1 # + loss2 + loss3

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        
        return output