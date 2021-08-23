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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

from GraphRec import *
from preprocessing import preprocess #RecoDataset, MyCollator, collate_fn

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


##training

def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        #if i % 100 == 0:
        #    print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
        #        epoch, i, running_loss / 100, best_rmse, best_mae))
        #    running_loss = 0.0
    return running_loss


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_mse = mean_squared_error(tmp_pred, target)
    mae = mean_absolute_error(tmp_pred, target)
    expected_rmse = mean_squared_error(expected_mse)
    return expected_mse, expected_rmse, mae


if __name__ == "__main__":

    dir_data = './data/'
    path_data = args.datadir+args.dataset+"_dataset.pickle"

    if not os.path.exists(path_data):
        preprocess(args.datadir, args.dataset,args.test)

    
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, val_u, val_v, val_r, social_adj_lists, ratings_list = pickle.load(
        data_file)

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()+1

    print(num_ratings)

    print(min(train_u),max(train_u))
    print(min(train_v),max(train_v))

    print(min(list(history_u_lists.keys())), max(list(history_u_lists.keys())))
    print(min(list(social_adj_lists.keys())), max(list(social_adj_lists.keys())))

    print(len(train_u), len(list(social_adj_lists.keys())))
    print(len(np.intersect1d(list(history_u_lists.keys()), list(social_adj_lists.keys()))))
    print(len( list(social_adj_lists.keys())),  len(list(history_u_lists.keys())))

    
    #data splitting 

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))

    valset = torch.utils.data.TensorDataset(torch.LongTensor(val_u), torch.LongTensor(val_v),
                                             torch.FloatTensor(val_r))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,drop_last=True )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=True)

    #model initialization
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    #positionnal encodings initialization for users, items and ratings    
    u2e = nn.Embedding(num_users, args.embedding_dim).to(device)
    v2e = nn.Embedding(num_items, args.embedding_dim).to(device)
    r2e = nn.Embedding(num_ratings, args.embedding_dim).to(device)

    model = GraphRec(u2e, v2e, r2e,  history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,social_adj_lists, embedding_dim = args.embedding_dim, device = device).to(device) #device,num_users,num_items,num_ratings,args.embedding_dim).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)

    criterion = nn.MSELoss()

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):

        tr_loss = train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        train_losses.append(tr_loss)

        # please add the validation set to tune the hyper-parameters based on your datasets.
        val_loss , val_rmse, val_mae = test(model, device, val_loader)

        # early stopping 
        if best_rmse > val_rmse:
            best_rmse = val_rmse
            best_mae = val_mae
            endure_count = 0
        else:
            endure_count += 1
        print("Epoch %d , training loss: %.4f,  val loss: %.4f, val rmse: %.4f, val mae:%.4f " % (tr_loss, val_loss, val_rmse, val_mae))

        if endure_count > 5:
            break


        """
        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        save_path = '%soutput/%s/%s.result' % (args['results_path'], args['dataset'])
        f = open(save_path, 'a')

        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
            % (args['embed_size'], 0.0001, args['layer_size'], args['node_dropout'], args['mess_dropout'], args['regs'],
            args['adj_type'], final_perf))
        
        f.close()
        """
    

    