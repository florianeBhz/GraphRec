import torch
import torch.nn as nn
import pickle
import numpy as np
import random
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import torch.nn.utils.rnn as rnn 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='epinion', help='dataset name: ciao/epinion')
parser.add_argument('--datadir', default='data/', help='data directory')

parser.add_argument('--test', default=0.1, help='test proportion')
args = parser.parse_args()



def preprocess():
    ratings_file = args.datadir+args.dataset+"_rating_with_timestamp.txt"
    trust_file = args.datadir+args.dataset+"_trust.txt"
    
    click_f = np.loadtxt(ratings_file, dtype = np.int32)
    trust_f = np.loadtxt(trust_file, dtype = np.int32)


    click_list = []
    trust_list = []

    u_items_list = []
    u_users_list = []
    # 3D list : for each user==> for each of its neighbours ==>list of items : [ u1: [ n1:[i1,i2], n2:[i2,i5]] , u2: [ n1:[i3],n2:[i1]] , ... , un: [ [],[]]]
    u_users_items_list = []
    i_users_list = []

    u_items_ratings = []
    i_users_ratings = []
    u_users_items_ratings = []

    user_count = 0
    item_count = 0
    rate_count = 0

    for s in click_f:
        uid = s[0]
        iid = s[1]
        if args.dataset == 'ciao':
            label = s[3]
        elif args.dataset == 'epinion':
            label = s[2]

        if uid > user_count:
            user_count = uid
        if iid > item_count:
            item_count = iid
        if label > rate_count:
            rate_count = label
        click_list.append([uid, iid, label])


    #social adj
    for s in trust_f:
        uid = s[0]
        fid = s[1]
        if uid > user_count or fid > user_count:
            continue
        trust_list.append([uid, fid])

    trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])
    trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')

    print(user_count,item_count,rate_count)

    pos_list = []
    for i in range(len(click_list)):
        pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

    # remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
    #pos_list = list(set(pos_list))

    print("Statistics \n")
    print("# interactions :",str(len(pos_list)))
    df = pd.DataFrame(pos_list, columns = ['uid', 'iid', 'label'])

    users_with_social = np.unique(trust_df.to_numpy().flatten())
    users_with_interact = np.intersect1d(np.unique(df['uid']),users_with_social)

    df = df[df.uid.isin(users_with_interact)]
    
    print("# users :", str(len(users_with_interact)))  #18088 users
    print("# items :", len(np.unique(df['iid'])))   #261649 items
    print("# interactions :", str(df.shape)) #764352 ratings



    # train, valid and test data split
    df = shuffle(df)
    num_test = int(len(df) * args.test)
    test_set = df.iloc[:num_test]
    valid_set = df.iloc[num_test:2 * num_test]
    train_set = df.iloc[2 * num_test:]
    print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))

    
    with open(args.datadir + '/dataset.pkl', 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    

    train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
    valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
    test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

    click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
    train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')

    print(len(train_df.uid.unique()))
        


    for u in tqdm(range(user_count + 1)):
        history_u = train_df[train_df['uid'] == u]
        u_items = history_u['iid'].tolist()
        u_ratings = history_u['label'].tolist()
        if u_items == []:
            u_items_list.append([0])
            u_items_ratings.append([0])
        else:
            u_items_list.append(u_items)
            u_items_ratings.append(u_ratings)


    train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

    
    for i in tqdm(range(item_count + 1)):
        history_v = train_df[train_df['iid'] == i]
        i_users = history_v['uid'].tolist()
        i_ratings = history_v['label'].tolist()
        if i_users == []:
            i_users_list.append([0])
            i_users_ratings.append([0])
        else:
            i_users_list.append(i_users)
            i_users_ratings.append(i_ratings)

    
    for u in tqdm(range(user_count + 1)):
        social_adj = trust_df[trust_df['uid'] == u]
        u_users = social_adj['fid'].unique().tolist()
        if u_users == []:
            u_users_list.append([0])
            u_users_items_list.append([[0]])
            u_users_items_ratings.append([[0]])
        else:
            u_users_list.append(u_users)
            uu_items = []
            uu_ratings = []
            for uid in u_users:
                uu_items.append(u_items_list[uid])
                uu_ratings.append(u_items_ratings[uid])
            u_users_items_list.append(uu_items)
            u_users_items_ratings.append(uu_ratings)

 

class RecoDataset(Dataset):
    """Personnalized dataset format"""

    def __init__(self, dataset, list_file,root_dir, transform=None):
        """
        Args:
            dataset (df): train | val |test 
            list_file (string): Path to the file which contains lists of users items, users social adj, ratings , ... 
            root_dir (string): Directory of data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = dataset
        self.root_dir = root_dir
        self.transform = transform

        with open(self.root_dir + list_file, 'rb') as f:
            self.u_items_list = pickle.load(f)
            self.u_users_list = pickle.load(f)
            self.u_items_ratings = pickle.load(f)
            self.i_users_ratings = pickle.load(f)
            self.u_users_items_list = pickle.load(f)
            self.u_users_items_ratings = pickle.load(f)
            self.i_users_list = pickle.load(f)
            self.user_count, self.item_count, self.rate_count = pickle.load(f)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        uid, iid, rt = self.data[idx][0], self.data[idx][1], self.data[idx][2]
        u_items = self.u_items_list[uid]
        u_users = self.u_users_list[uid]
        u_items_r = self.u_items_ratings[uid]
        i_users_r = self.i_users_ratings[iid]
        u_users_items = self.u_users_items_list[uid]
        u_users_items_r = self.u_users_items_ratings[uid]
        i_users= self.i_users_list[iid]
        

        sample = {'uid': uid, 'iid': iid, 'rt':rt, 'u_items':u_items, 'u_users':u_users, 'u_users_items':u_users_items, 
        'i_users':i_users, 'u_items_r':u_items_r , 'i_users_r':i_users_r, 'u_users_items_r':u_users_items_r}

        if self.transform:
            sample = self.transform(sample)

        return sample



def MyCollator(batch):

    u_max_cut_lenght = 30
    i_max_cut_lenght = 15
    uu_max_cut_lenght = 3

    #extracting sequences of u_items, u_users for batch elem
    u_items_batch = list(map(lambda elm: torch.tensor(elm['u_items']),batch))
    u_users_batch = list(map(lambda elm: torch.tensor(elm['u_users']),batch))
    i_users_batch = list(map(lambda elm: torch.tensor(elm['i_users']),batch))
    u_items_r_batch = list(map(lambda elm: torch.tensor(elm['u_items_r']),batch))
    i_users_r_batch = list(map(lambda elm: torch.tensor(elm['i_users_r']),batch))

    #padding with maximum len
    u_items_padded = rnn.pad_sequence(u_items_batch, batch_first=True, padding_value=0)
    u_users_padded = rnn.pad_sequence(u_users_batch, batch_first=True, padding_value=0)
    i_users_padded = rnn.pad_sequence(i_users_batch, batch_first=True, padding_value=0)
    u_items_r_padded = rnn.pad_sequence(u_items_r_batch, batch_first=True, padding_value=0)
    i_users_r_padded = rnn.pad_sequence(i_users_r_batch, batch_first=True, padding_value=0)


    #u_user_items and u_users_items_r

    print(torch.tensor(u_items_padded).shape)

    print(torch.tensor(u_items_padded)[:,:u_max_cut_lenght])
    """
    
    for elm in batch:
        #padding
        print(elm['u_items'])
        u_items_pad = torch.tensor(u_items_pad)
        #u_users_pad = torch.tensor(u_users_pad)
        print(u_items_pad.shape)

    """

def main():

    preprocess()
        

    
if __name__ == "__main__":
    
    main()