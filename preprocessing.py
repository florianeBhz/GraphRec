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

truncate_len = 30


def preprocess(datadir,dataset,test):
    ratings_file = datadir+dataset+"_rating_with_timestamp.txt"
    trust_file = datadir+dataset+"_trust.txt"
    
    click_f = np.loadtxt(ratings_file, dtype = np.int32)
    trust_f = np.loadtxt(trust_file, dtype = np.int32)

    click_list = []
    trust_list = []

    u_items_list = [] #history_u_lists
    u_users_list = [] #social_adj_lists

    # 3D list : for each user==> for each of its neighbours ==>list of items : [ u1: [ n1:[i1,i2], n2:[i2,i5]] , u2: [ n1:[i3],n2:[i1]] , ... , un: [ [],[]]]
    u_users_items_list = []
    i_users_list = [] #history_v_lists

    u_items_ratings = []  #history_ur_lists
    i_users_ratings = []  #history_vr_lists
    u_users_items_ratings = []

    user_count = 0
    item_count = 0
    rate_count = 0


    for s in click_f:
        uid = s[0]
        iid = s[1]
        label = s[3]

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
    num_test = int(len(df) * test)
    test_set = df.iloc[:num_test]
    valid_set = df.iloc[num_test:2 * num_test]
    train_set = df.iloc[2 * num_test:]
    print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))


    train_u, train_v, train_r = train_set.to_numpy()[:,0],train_set.to_numpy()[:,1],train_set.to_numpy()[:,2]
    
    test_u, test_v, test_r = test_set.to_numpy()[:,0],test_set.to_numpy()[:,1],test_set.to_numpy()[:,2]

    val_u, val_v, val_r = valid_set.to_numpy()[:,0],valid_set.to_numpy()[:,1],valid_set.to_numpy()[:,2]

    
    train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
    valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
    test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

    click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
    train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')
                                                        

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
        
    history_u_lists = dict(zip(range(len(u_items_list)), u_items_list))
    history_ur_lists = dict(zip(range(len(u_items_ratings)), u_items_ratings))

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

    history_v_lists = dict(zip(range(len(i_users_list)), i_users_list))
    history_vr_lists = dict(zip(range(len(i_users_ratings)), i_users_ratings))

    
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

    social_adj_lists = dict(zip(range(len(u_users_list)), u_users_list))

    ratings_list = np.arange(1,rate_count+1,1) #ratings_list = np.arange(0,rate_count+0.5,0.5)
    ratings_list = dict(zip(ratings_list, range(len(ratings_list))) ) #{2.0: 0, 1.0: 1, 3.0: 2, 4.0: 3, 2.5: 4, 3.5: 5, 1.5: 6, 0.5: 7}

    to_save = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, val_u, val_v, val_r, social_adj_lists, ratings_list]
    
    pickle.dump( to_save, open( datadir+dataset+"_dataset.pickle", "wb" ) )





def preprocess_old():
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
            label = s[3]

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

    with open(args.datadir + '/list.pkl', 'wb') as f:
        pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_items_ratings, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_users_ratings, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(u_users_items_ratings, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)


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
        'i_users':i_users, 'u_items_r':u_items_r , 'u_users_items_r':u_users_items_r,'i_users_r':i_users_r,}


        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, iids, labels = [], [], []
    u_items, u_users, u_users_items, i_users = [], [], [], []
    u_items_r ,u_users_items_r,i_users_r = [], [], []
    u_items_len, u_users_len, i_users_len = [], [], []
    u_items_r_len = []
    for data in batch_data:

        uid, iid, label, u_items_u, u_users_u, u_users_items_u, i_users_i, u_items_r_u ,  u_users_items_r_u, i_users_r_i = list(data.values()) 

        uids.append(uid)
        iids.append(iid)
        labels.append(label)

        # user-items    
        if len(u_items_u) <= truncate_len:
            u_items.append(u_items_u)
            u_items_r.append(u_items_r_u)
        else:
            u_items.append(random.sample(u_items_u, truncate_len))
            u_items_r.append(random.sample(u_items_r_u, truncate_len))
        u_items_len.append(min(len(u_items_u), truncate_len))
        
        # user-users and user-users-items
        if len(u_users_u) <= truncate_len:
            u_users.append(u_users_u)
            u_u_items = []
            u_u_items_r = [] 
            for uuid, uui in enumerate(u_users_items_u):
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                    u_u_items_r.append(u_users_items_r_u[uuid])
            
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
                    u_u_items_r.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)
            u_users_items_r.append(u_u_items_r)
        else:
            sample_index = random.sample(list(range(len(u_users_u))), truncate_len)
            u_users.append([u_users_u[si] for si in sample_index])

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = [] 
            u_u_items_r = []
            for uuid,uui in enumerate(u_users_items_u_tr):
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                    u_u_items_r.append(u_users_items_r_u[uuid])
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
                    u_u_items_r.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)
            u_users_items_r.append(u_u_items_r)

        u_users_len.append(min(len(u_users_u), truncate_len))	

        # item-users
        if len(i_users_i) <= truncate_len:
            i_users.append(i_users_i)
            i_users_r.append(i_users_r_i)
        else:
            i_users.append(random.sample(i_users_i, truncate_len))
            i_users_r.append(random.sample(i_users_r_i, truncate_len))
        i_users_len.append(min(len(i_users_i), truncate_len))

    batch_size = len(batch_data)

    # padding
    u_items_maxlen = max(u_items_len)
    u_users_maxlen = max(u_users_len)
    i_users_maxlen = max(i_users_len)
    
    u_item_pad = torch.zeros([batch_size, u_items_maxlen], dtype=torch.long)
    u_item_r_pad = torch.zeros([batch_size, u_items_maxlen], dtype=torch.long)

    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui)] = torch.LongTensor(ui)
    for i, uir in enumerate(u_items_r):
        u_item_r_pad[i, :len(uir)] = torch.LongTensor(uir)    
    
    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)
    
    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen], dtype=torch.long)
    u_user_item_r_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui)] = torch.LongTensor(ui)
    
    for i, uu_items_r in enumerate(u_users_items_r):
        for j, uir in enumerate(uu_items_r):
            #print(u_user_item_r_pad.shape,torch.LongTensor(uir).shape)
            u_user_item_r_pad[i, j, :len(uir)] = torch.LongTensor(uir)[:min(torch.LongTensor(uir).shape[0],truncate_len)]


    i_user_pad = torch.zeros([batch_size, i_users_maxlen], dtype=torch.long)
    i_user_r_pad = torch.zeros([batch_size, i_users_maxlen], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu)] = torch.LongTensor(iu)

    for i, iur in enumerate(i_users_r):
        i_user_r_pad[i, :len(iur)] = torch.LongTensor(iur)

    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(labels), \
            u_item_pad, u_user_pad, u_user_item_pad, i_user_pad, u_item_r_pad,u_user_item_r_pad, i_user_r_pad


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


    print(torch.tensor(u_items_padded).shape)
    print(torch.tensor(u_items_padded)[:,:u_max_cut_lenght])
    
    # user-users and user-users-items
    if len(u_users_u) <= truncate_len:
        u_users.append(u_users_u)
        u_u_items = [] 
        for uui in u_users_items_u:
            if len(uui) < truncate_len:
                u_u_items.append(uui)
            else:
                u_u_items.append(random.sample(uui, truncate_len))
        u_users_items.append(u_u_items)
    else:
        sample_index = random.sample(list(range(len(u_users_u))), truncate_len)
        u_users.append([u_users_u[si] for si in sample_index])

        u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
        u_u_items = [] 
        for uui in u_users_items_u_tr:
            if len(uui) < truncate_len:
                u_u_items.append(uui)
            else:
                u_u_items.append(random.sample(uui, truncate_len))
        u_users_items.append(u_u_items)

    u_users_len.append(min(len(u_users_u), truncate_len))	



def main():
    preprocess()
        
    
if __name__ == "__main__":
    
    main()