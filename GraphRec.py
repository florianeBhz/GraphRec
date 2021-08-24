import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP,self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.layers(x)


#a generic attention class for user , social and item aggregation, ex of user aggregation
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        #input dim is 2* embedding dim because of concatenation
        self.att1 = nn.Linear(self.input_dim , self.input_dim//2)
        self.att2 = nn.Linear(self.input_dim //2, self.input_dim //2)
        self.att3 = nn.Linear(self.input_dim //2, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, neighs_reps, u_rep, num_neighs):
        #we repeat self node representation num_neighs times
        pi_reps = u_rep.repeat(num_neighs, 1)
        #we augment each neighbor repr with the self node repr
        alpha_ia_star = torch.cat((neighs_reps, pi_reps), -1)
        #
        alpha_ia_star = F.relu(self.att1(alpha_ia_star))
        alpha_ia_star = F.dropout(alpha_ia_star, training=self.training)
        alpha_ia_star = F.relu(self.att2(alpha_ia_star)) 
        alpha_ia_star = F.dropout(alpha_ia_star, training=self.training)
        alpha_ia_star = self.att3(alpha_ia_star)#Eq 5
        alpha_ia = F.softmax(alpha_ia_star, dim=0) #Eq 6
        return alpha_ia    
    
#perform user modeling in two steps for batched users
class UserModeling(nn.Module):
    def __init__(self,device,embed_u, embed_i, embed_r,social_adj_lists, embed_dim):
        super(UserModeling, self).__init__()

        self.device = device
        self.embed_u = embed_u
        self.embed_i = embed_i
        self.embed_r = embed_r
        self.social_adj_lists = social_adj_lists
        self.embed_dim = embed_dim

        self.gv = MLP(2*self.embed_dim,self.embed_dim)
        self.att_I = Attention(2*self.embed_dim)
        self.att_S = Attention(2*self.embed_dim)

        self.mlp = MLP(2*self.embed_dim,self.embed_dim)


    def forward(self,nodes_u,history_u_lists_batch, social_adj_lists_batch,history_ur_lists_batch):
        
        #matrix of batched users item aggregation vectors
        embed_matrix_I = torch.empty(len(history_u_lists_batch), self.embed_dim, dtype=torch.float).to(self.device)
        #matrix of batched users social aggregation vectors
        embed_matrix_S = torch.empty(len(nodes_u), self.embed_dim, dtype=torch.float).to(self.device)

        #for each user in the batch
        for i in range(len(history_u_lists_batch)): #user items lists

            #item aggregation
            i_v_list = history_u_lists_batch[i]
            i_v_list_len = len(i_v_list)
            i_v_label = history_ur_lists_batch[i]

            qa = self.embed_i.weight[list(i_v_list)]
            pi = self.embed_u.weight[nodes_u[i]]
            er = self.embed_r.weight[list(i_v_label)]

            #opinion-aware representation
            xia = self.gv(torch.cat((qa,er),1))  #Eq 2      

            #computing attention coefficeients for item aggregation
            alpha_ia = self.att_I(xia, pi, i_v_list_len) #Eq 5, 6

            #Item aggregation
            hi_I = torch.mm(xia.t(), alpha_ia) #Eq 4

            hi_I = hi_I.t()

            embed_matrix_I[i] = hi_I

            ## Social aggregation

            #getting the user social neighbors
            tmp_adj = self.social_adj_lists[i]
            num_neighs_u = len(tmp_adj)

            u_neighs_emb = self.embed_u.weight[list(tmp_adj)] 

            #computing attention coefficeients
            beta_io = self.att_S(u_neighs_emb, pi, num_neighs_u) #Eq 10, 11

            #Social aggregation vector

            hi_S = torch.mm(u_neighs_emb.t(), beta_io).t() #Eq 8
            
            embed_matrix_S[i] = hi_S

        #combining item aggregation and social aggregation
        c1 =  torch.cat((embed_matrix_I,embed_matrix_S),1) #Eq 12

        hi = self.mlp(c1) #Eq 13, 14

        return hi


#perform item modeling for batched items
class ItemModeling(nn.Module):
    def __init__(self,device,embed_i, embed_u, embed_r, embed_dim):
        super(ItemModeling, self).__init__()

        self.device = device
        self.embed_i = embed_i
        self.embed_u = embed_u
        self.embed_r = embed_r
        self.embed_dim = embed_dim

        self.gu = MLP(2*self.embed_dim,self.embed_dim)
        self.att = Attention(2*self.embed_dim) 
                

    def forward(self,nodes_v,history_v_lists_batch,history_vr_lists_batch):

        embed_matrix = torch.empty(len(history_v_lists_batch), self.embed_dim, dtype=torch.float).to(self.device)

        #for each item
        for j in range(len(history_v_lists_batch)): #items neighbors lists
            #get user that intracted with item j
            j_u_list = history_v_lists_batch[j]
            j_u_list_len = len(j_u_list)
            #get ratings given by users which interacted with j
            j_u_label = history_vr_lists_batch[j]

            #item j users (neighbors) embeddings
            pt = self.embed_u.weight[list(j_u_list)] 
            #item j embedding
            qj = self.embed_i.weight[nodes_v[j]]
            #ratings embeddings for item j by users 
            er = self.embed_r.weight[list(j_u_label)]

            #opinion-aware representation
            fjt = self.gu(torch.cat((pt,er),1)) #Eq 15

            #attention coefficients for aggregation
            mu_jt = self.att(fjt, qj, j_u_list_len) #Eq 18 et 19

            #user aggregated vector
            zj = torch.mm(fjt.t(), mu_jt).t() #Eq 17

            embed_matrix[j] = zj

        return embed_matrix



class GraphRec(nn.Module):
    def __init__(self, u2e, v2e, r2e, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, embedding_dim, device):
        super(GraphRec, self).__init__()


        self.device = device
        self.embed_u = u2e
        self.embed_i = v2e
        self.embed_r = r2e
        self.embedding_dim = embedding_dim
        
        #users items list
        self.history_u_lists = history_u_lists
        #users items ratings list
        self.history_ur_lists = history_ur_lists 
        #item users list
        self.history_v_lists = history_v_lists
        #item users ratings list
        self.history_vr_lists = history_vr_lists
        #users social network list
        self.social_adj_lists = social_adj_lists


        self.w_ur1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.w_ur2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.w_vr1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.w_vr2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.w_uv1 = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.w_uv2 = nn.Linear(self.embedding_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embedding_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embedding_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

        #initializing user modeling model
        self.model_user = UserModeling(self.device,self.embed_u, self.embed_i, self.embed_r,self.social_adj_lists, self.embedding_dim).to(self.device)
        #initializing item modeling model
        self.model_item = ItemModeling(self.device,self.embed_i, self.embed_u, self.embed_r,self.embedding_dim).to(self.device)

        self.g = MLP(2*self.embedding_dim,1) 

    def forward(self, nodes_u,nodes_v):

        history_u_lists_batch = []
        history_ur_lists_batch = []
        history_v_lists_batch = []
        history_vr_lists_batch = []
        social_adj_lists_batch = []

        #for each user in the batch
        for u in nodes_u:
            #retrieve user's items list, ratings list, and social network
            history_u_lists_batch.append(self.history_u_lists[int(u)])
            history_ur_lists_batch.append(self.history_ur_lists[int(u)])

            social_adj_lists_batch.append(self.social_adj_lists[int(u)])

        #for each item in the batch
        for v in nodes_v:
            #retrieve item's users list, ratings list
            history_v_lists_batch.append(self.history_v_lists[int(v)])
            history_vr_lists_batch.append(self.history_vr_lists[int(v)])

        #get user modeling vector
        hi = self.model_user(nodes_u,history_u_lists_batch,social_adj_lists_batch,history_ur_lists_batch) 

        #get item modeling vector 
        zj = self.model_item(nodes_v,history_v_lists_batch, history_vr_lists_batch) 

        hi = F.relu(self.bn1(self.w_ur1(hi)))
        hi = F.dropout(hi, training=self.training)
        hi = self.w_ur2(hi)

        zj = F.relu(self.bn2(self.w_vr1(zj)))
        zj = F.relu(self.w_vr1(zj))
        zj = F.dropout(zj, training=self.training)
        zj = self.w_vr2(zj)

        #concat the two representations
        x = torch.cat((hi, zj), 1) #Eq 20
        x = F.relu(self.bn3(self.w_uv1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x) #Eq 23
        return scores.squeeze()



    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)

