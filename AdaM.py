import torch
from network import AdaM_MVC
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss, ClusterCo
from dataloader import load_data
import os
import pandas as pd
import torch.nn.functional as F


def get_args(Dataname):
    print(Dataname)
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--temperature_l", default=1.0)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--rec_epochs", default=200)
    parser.add_argument("--fine_tune_representation_epochs", default=0)
    parser.add_argument("--fine_tune_structure_epochs", default=100)
    parser.add_argument("--low_feature_dim", default=512)
    parser.add_argument("--high_feature_dim", default=128)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "DHA":
        args.fine_tune_structure_epochs = 100
        seed = 1
    if args.dataset == "Caltech":
        args.fine_tune_structure_epochs = 200
        seed = 5
    if args.dataset == "NUSWIDE":
        args.fine_tune_structure_epochs = 100
        seed = 1
    if args.dataset == "YoutubeVideo":
        args.fine_tune_structure_epochs = 100
        seed = 5
    if args.dataset == "Sources3":
        args.fine_tune_structure_epochs = 10
        args.batch_size = 100
        seed = 5
    if args.dataset == "coil20":
        args.fine_tune_structure_epochs = 110
        seed = 10
    if args.dataset == "MNIST-USPS":
        args.fine_tune_structure_epochs = 100
        seed = 10
    if args.dataset == "Fashion":
        args.fine_tune_structure_epochs = 100
        seed = 10
    if args.dataset == "CCV":
        args.fine_tune_structure_epochs = 100
        seed = 3
    if args.dataset == "Hdigit":
        args.fine_tune_structure_epochs = 100
        seed = 10
    if args.dataset == "Cifar10":
        args.fine_tune_structure_epochs = 100
        seed = 10
    if args.dataset == "Cifar100":
        args.fine_tune_structure_epochs = 200
        seed = 10
    if args.dataset == "Caltech-2V":
        args.fine_tune_structure_epochs = 100
        seed = 10
    if args.dataset == "Caltech-3V":
        args.fine_tune_structure_epochs = 100
        seed = 10
    if args.dataset == "Caltech-4V":
        args.fine_tune_structure_epochs = 150
        seed = 10
    if args.dataset == "Caltech-5V":
        args.fine_tune_structure_epochs = 200
        seed = 5
    if args.dataset == "Caltech20":
        args.fine_tune_structure_epochs = 200
        seed = 5
    if args.dataset == "Caltech7":
        args.fine_tune_structure_epochs = 200
        seed = 5
    if args.dataset == "NGs":
        args.fine_tune_structure_epochs = 100
        seed = 10

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(seed)

    dataset, dims, view, data_size, class_num= load_data(args.dataset)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    return args,device,dataset, dims, view, data_size, class_num,data_loader

def pre_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _,_ = model(xs)
        loss_list = []
        for v in range(view):
            rl = mse(xs[v], xrs[v])
            loss_list.append(rl)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


###结构引导的对比损失+聚类级别的对比损失+自适应对比学习
def fine_tune_structure(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs, qls = model(xs)
        commonz, commonz_qhs,S,qhs,qhsf,weights, SV, commonz_fused_view = model.AdaM(xs)

        sorted_indices = torch.argsort(weights, descending=False) 
        sorted_weights = sorted(weights, reverse=True)
        r = 2 / (view * (view - 1))
        half_views = len(sorted_indices) // 2
        if(len(sorted_indices) % 2 == 1): 
            num_views_to_update = half_views + 1
        else:
            num_views_to_update = half_views
        weight_big = sum(sorted_weights[:num_views_to_update])*r
        weight_small = (1 - sum(sorted_weights[:num_views_to_update]))*r
        loss_list = []
        for v in range(view):
            if sorted_indices[v] < num_views_to_update:                                  
                loss_list.append(weight_big * criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S))
                loss_list.append(weight_big * criterion2.forward(qls[v], commonz_qhs))
                loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(weight_big *mes(zs[v], SV[v]))
            else:
                loss_list.append(weight_small * criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S))
                loss_list.append(weight_small * criterion2.forward(qls[v], commonz_qhs))
                loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(weight_small *mes(zs[v], SV[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return tot_loss/len(data_loader),weights


    
if __name__ == '__main__':
    if not os.path.exists('./models'):
        os.makedirs('./models')
    Datalist = ['Sources3'] 
    for Dataname in Datalist:
        args, device,dataset, dims, view, data_size, class_num,data_loader = get_args(Dataname)  
        ##训练  
        model = AdaM_MVC(view, dims, class_num,args.low_feature_dim, args.high_feature_dim, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
        criterion2 = ClusterCo(view,args.batch_size, class_num, args.temperature_l, device).to(device)
        epoch = 1
        while epoch <= args.rec_epochs:
            pre_train(epoch)
            epoch += 1
            
        while epoch <= args.rec_epochs + args.fine_tune_structure_epochs:
            fine_loss,weights = fine_tune_structure(epoch)
            if epoch == args.rec_epochs  + args.fine_tune_representation_epochs + args.fine_tune_structure_epochs:
                print('---test---',weights)
                acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num)
                state = model.state_dict()
                torch.save(state, './models/' + args.dataset  +'.pth')
                print('Saving model...')
                print("successful:",Dataname)
            epoch += 1
       



