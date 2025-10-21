import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        #h_i是特定view，h_j是common
        S = torch.pow(S, 2)
        S_1 = S.repeat(2, 2)
        all_one = torch.ones(self.batch_size*2, self.batch_size*2).to('cuda')
        S_2 = all_one - S_1
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim1 = torch.multiply(sim, S_2)          
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    
class ClusterCo(nn.Module):
    def __init__(self, view, batch_size, class_num, temperature_f, device):
        super(ClusterCo, self).__init__()
        self.class_num = class_num
        self.temperature = temperature_f
        self.n_views = view

        self.mask = self.mask_correlated_clusters(self.class_num)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_cluster(self, c_i,c_j):
        p_i = torch.mean(c_i, dim=0)
        ne_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = torch.mean(c_j, dim=0)
        ne_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = F.cosine_similarity(c.unsqueeze(1), c.unsqueeze(0), dim=2) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = F.cross_entropy(logits, labels)

        #return loss, ne_loss
        return loss

    def forward(self,cu,cv):
        losses = self.forward_cluster(cu, cv)             
        return losses
