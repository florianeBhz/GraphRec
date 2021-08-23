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

    ratings_list = np.arange(0.5,rate_count+0.5,0.5) #np.arange(1,rate_count+1,1) #ratings_list = np.arange(0,rate_count+0.5,0.5)
    ratings_list = dict(zip(ratings_list, range(len(ratings_list))) ) #{2.0: 0, 1.0: 1, 3.0: 2, 4.0: 3, 2.5: 4, 3.5: 5, 1.5: 6, 0.5: 7}

    to_save = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, val_u, val_v, val_r, social_adj_lists, ratings_list]
    
    pickle.dump( to_save, open( datadir+dataset+"_dataset.pickle", "wb" ) )



def main():
    preprocess()
        
    
if __name__ == "__main__":
    
    main()