import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch


__all__ = ['DomainAdversarialLoss']


class Class_Domaindiscri(nn.Module):
    def __init__(self, embedding_dim: int, num_domains: int, num_class : int):
        super(Class_Domaindiscri, self).__init__()
        self.num_class = num_class
        self.cls_discri = []

        for i in range(num_class):
            #self.cls_discri.append(nn.Linear(embedding_dim, num_domains).cuda())
            self.cls_discri.append(
            nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_domains)
            ).cuda())

    def forward(self, f_s, f_t):
        """"""
        f_s = f_s.detach()
        f_t = f_t.detach()
        pred_d_t = []
        pred_d_s = []
        for i in range(self.num_class):
            pred_d_t_i = self.cls_discri[i](f_t)
            pred_d_t.append(pred_d_t_i)
            pred_d_s_i = self.cls_discri[i](f_s)
            pred_d_s.append(pred_d_s_i)
        cd_t = torch.stack(pred_d_t,dim=1).cuda()
        cd_s = torch.stack(pred_d_s, dim=1).cuda()

        return cd_s, cd_t

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = []
        for i in range(self.num_class):
            params.append({"params": self.cls_discri[i].parameters(), "lr_mult": 1})
        return params



def rho_pred(y_t,cd_t):
    cd_t = nn.Softmax(dim=2)(cd_t.detach())
    rho_cd = torch.mul(y_t,cd_t[:,:,0])
    return rho_cd


def binary_CE(cd_s,cd_t,rho_t,labels_s):
    d_label_s = torch.full((cd_s.size(0), 1),1).squeeze().cuda()
    d_label_t = torch.full((cd_t.size(0), 1),0).squeeze().cuda()
    rho_t = rho_t.detach()

    bce_loss_s = 0
    symbol= torch.full_like(cd_s[0],0).cuda()
    for i in range(cd_s.size(1)):
        for j in range(cd_s.size(0)):
            if labels_s[j].item() == i:
                symbol=1
            else:
                symbol=0
        bce_loss_s += torch.mean(symbol * F.cross_entropy(cd_s[:, i, :], d_label_s, reduction='none'))
    bce_loss_s = bce_loss_s/cd_s.size(1)

    bce_loss_t = 0
    for i in range(cd_t.size(1)):
        bce_loss_t += torch.mean(rho_t[:,i]*F.cross_entropy(cd_t[:,i,:],d_label_t,reduction='none'))
    bce_loss_t = bce_loss_t/cd_t.size(1)

    return (bce_loss_s+bce_loss_t)/2











