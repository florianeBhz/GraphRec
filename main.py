import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.utils.rnn as rnn 

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from GraphRec import *
from preprocessing import RecoDataset, MyCollator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='epinion', help='dataset name: ciao/epinion')
parser.add_argument('--datadir', default='data/', help='data directory')
parser.add_argument('--test', default=0.1, help='test proportion')
parser.add_argument('--embedding_dim', default=64, help='embedding size')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
args = parser.parse_args()



with open(args.datadir + 'dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f).to_numpy()
    val_dataset = pickle.load(f).to_numpy() 
    test_dataset = pickle.load(f).to_numpy()

with open(args.datadir + 'list.pkl', 'rb') as f:
    u_items_list = pickle.load(f)
    u_users_list = pickle.load(f)
    u_items_ratings = pickle.load(f)
    i_users_ratings = pickle.load(f)
    u_users_items_list = pickle.load(f)
    u_users_items_ratings = pickle.load(f)
    i_users_list = pickle.load(f)
    (user_count, item_count, rate_count) = pickle.load(f)



#data loading
trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_dataset[:,0]), torch.LongTensor(train_dataset[:,1]),
                                              torch.FloatTensor(train_dataset[:,2]))
val_dataset = torch.utils.data.TensorDataset(torch.LongTensor(val_dataset[:,0]), torch.LongTensor(val_dataset[:,1]),
                                              torch.FloatTensor(val_dataset[:,2]))
test_dataset = torch.utils.data.TensorDataset(torch.LongTensor(test_dataset[:,0]), torch.LongTensor(test_dataset[:,1]),
                                              torch.FloatTensor(test_dataset[:,2]))


trainset_reco = RecoDataset(train_dataset,'list.pkl','data/')
val_set_reco = RecoDataset(val_dataset,'list.pkl','data/')
test_set_reco = RecoDataset(test_dataset,'list.pkl','data/')

train_loader = DataLoader(trainset_reco, batch_size=args.batch_size, shuffle=True, collate_fn=MyCollator)
val_loader = DataLoader(val_set_reco, batch_size=args.batch_size, shuffle=False,collate_fn=MyCollator)
test_loader = DataLoader(test_set_reco, batch_size=args.test_batch_size, shuffle=False,collate_fn=MyCollator)


#model and optimizer initialization 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model = GraphRec(device,user_count,item_count,rate_count,args.embedding_dim).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)

loss = nn.MSELoss()


##training
def train(epoch):
    total_l = 0
    for data in train_loader:
        data = data.to(device)
        uids , iids, labels = data[:,0], data[:,1],data[:,2]
        
        (uids, iids, labels, u_items, u_users, u_users_items, i_users, u_items_r , i_users_r, u_users_items_r) = list(data.values())

       
        uids = torch.tensor(uids).to(device)
        iids = torch.tensor(iids).to(device)
        labels = torch.tensor(labels).to(device)
        u_items = torch.tensor(u_items).to(device)
        u_users = torch.tensor(u_users).to(device)
        u_users_items = torch.tensor(u_users_items).to(device)
        i_users = torch.tensor(i_users).to(device)
        i_users_r = torch.tensor(i_users_r).to(device)
        u_items_r = torch.tensor(u_items_r).to(device)
        u_users_items_r = torch.tensor(u_users_items_r).to(device)


        optimizer.zero_grad()
        outputs = model(uids, iids, u_items, u_users, i_users,u_items_r,u_users_items,u_users_items_r,i_users_r).squeeze()
        
        print(outputs[:3],labels[:3])

        lss = loss(outputs,labels.float() )

        lss.backward()
        optimizer.step()

        print(lss.cpu().detach().numpy())

        total_l += lss.cpu().detach().numpy()


if __name__ == "__main__":
    
    train(0)