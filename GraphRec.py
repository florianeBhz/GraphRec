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
            nn.Linear(output_dim,output_dim)
        )

    def forward(self,x):
        return self.layers(x)



"""
class Attention(nn.Module):
    def __init__(self,input_dim):
        super(Attention,self).__init__()

        self.input_dim = input_dim
        self.lin1 = nn.Linear(self.input_dim, self.input_dim//2,bias=True)
        self.lin2 = nn.Linear(self.input_dim//2, 1,bias=True)



    def forward(self,self_rep,neighs_rep):
        #print(self_rep.shape,num_neighs)
        pi_rep = self_rep.unsqueeze(1).expand_as(neighs_rep)
        #print(self_rep.shape, pi_rep.shape)
        alpha_ia_star = torch.cat((pi_rep,neighs_rep),-1)

        alpha_ia_star = F.relu(self.lin1(alpha_ia_star))
        alpha_ia_star = self.lin2(alpha_ia_star)

        alpha_ia = F.softmax(alpha_ia_star, dim=0)
        return alpha_ia
"""

class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, neighs_reps, u_rep, num_neighs):
        pi_reps = u_rep.repeat(num_neighs, 1)
        alpha_ia_star = torch.cat((neighs_reps, pi_reps), 1)
        alpha_ia_star = F.relu(self.att1(alpha_ia_star))
        alpha_ia_star = F.dropout(alpha_ia_star, training=self.training)
        alpha_ia_star = F.relu(self.att2(alpha_ia_star))
        alpha_ia_star = F.dropout(alpha_ia_star, training=self.training)
        alpha_ia_star = self.att3(alpha_ia_star)
        alpha_ia = F.softmax(alpha_ia_star, dim=0)
        return alpha_ia



class UserSocialSpace(nn.Module):
    def __init__(self,device,embed_u, embed_i, embed_r, embed_dim):
        super(UserSocialSpace, self).__init__()

        def forward(self,item_space_rep, history_):
            return 0


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
        self.att_O = Attention(2*self.embed_dim)
        self.att_S = Attention(2*self.embed_dim)

        self.linI = nn.Linear(self.embed_dim,self.embed_dim)
        self.linO = nn.Linear(self.embed_dim,self.embed_dim)
        self.linS = nn.Linear(self.embed_dim,self.embed_dim)

        self.mlp = MLP(2*self.embed_dim,self.embed_dim)



    def forward(self,nodes_u,history_u_lists_batch, u_users_list,history_ur_lists_batch):#,u_users_items,u_users_items_r,i_users_list,i_users_r):

        embed_matrix_I = torch.empty(len(history_u_lists_batch), self.embed_dim, dtype=torch.float).to(self.device)
        embed_matrix_S = torch.empty(len(nodes_u), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_u_lists_batch)): #user items lists
            i_v_list = history_u_lists_batch[i]
            i_v_list_len = len(i_v_list)
            i_v_label = history_ur_lists_batch[i]

            qa = self.embed_i(torch.LongTensor(i_v_list))
            pi = self.embed_u(torch.LongTensor([nodes_u[i]]))
            er = self.embed_r(torch.LongTensor([i_v_label]))

            xia = self.gv(torch.cat((qa,er),1))

            alpha_ia = self.att(xia, pi, i_v_list_len)

            ###

            hi_I = torch.mm(xia.t(), alpha_ia)

            hi_I = F.relu(self.w_r1(hi_I))

            hi_I = F.relu(self.w_r2(hi_I))

            hi_I = hi_I.t()

            embed_matrix_I[i] = hi_I

            ##

            tmp_adj = self.social_adj_lists[i]
            num_neighs_u = len(tmp_adj)

            u_neighs_emb = self.embed_u.weight[list(tmp_adj)] 

            att_w = self.att(u_neighs_emb, pi, num_neighs_u)
            att_history = torch.mm(u_neighs_emb.t(), att_w).t()
            embed_matrix_S[i] = att_history

            c1 =  torch.cat((embed_matrix_I,embed_matrix_S),1)

            hi = self.mlp(c1)
    
        return hi


class ItemModeling(nn.Module):
    def __init__(self,device,embed_i, embed_u, embed_r, embed_dim):
        super(ItemModeling, self).__init__()

        self.device = device
        self.embed_i = embed_i
        self.embed_u = embed_u
        self.embed_r = embed_r
        self.embed_dim = embed_dim

        self.gv = MLP(2*self.embed_dim,self.embed_dim)
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim) 
                

    def forward(self,nodes_v,history_v_lists_batch,history_vr_lists_batch):#i_idx,i_users_list,i_users_r):

        embed_matrix = torch.empty(len(history_v_lists_batch), self.embed_dim, dtype=torch.float).to(self.device)

        for j in range(len(history_v_lists_batch)): #items neighbors lists
            j_u_list = history_v_lists_batch[j]
            j_u_list_len = len(j_u_list)
            j_u_label = history_vr_lists_batch[j]

            pt = self.embed_u.weight[j_u_list]
            qj = self.embed_i.weight[nodes_v[j]]
            er = self.embed_r.weight[j_u_label]

            fjt = self.gu(torch.cat((pt,er),1))

            mu_jt = self.att(fjt, qj, j_u_list_len)

            ###

            zj = torch.mm(fjt.t(), mu_jt)

            zj = F.relu(self.w_r1(zj))

            zj = F.relu(self.w_r2(zj))

            zj = zj.t()

            embed_matrix[j] = zj

            """
            #user-aggregation
            qj = self.embed_i(i_idx)

            pt = self.embed_u(i_users_list[:,:])

            er = self.embed_r(i_users_r[:,:])

            #print(qj.shape,er.shape,pt.shape)

            zj = F.relu(self.lin(torch.sum(mu_jt*fjt,1)))
            """

            return zj



class GraphRec(nn.Module):
    def __init__(self, u2e, v2e, r2e, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, embedding_dim, device):
        super(GraphRec, self).__init__()


        self.device = device
        self.embed_u = u2e
        self.embed_i = v2e
        self.embed_r = r2e
        self.embedding_dim = embedding_dim
        
        self.history_u_lists = history_u_lists
        self.history_ur_lists = history_ur_lists 
        self.history_v_lists = history_v_lists
        self.history_vr_lists = history_vr_lists
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

        self.model_user = UserModeling(self.device,self.embed_u, self.embed_i, self.embed_r,self.social_adj_lists, self.embedding_dim).to(self.device)
        self.model_item = ItemModeling(self.device,self.embed_i, self.embed_u, self.embed_r,self.embedding_dim).to(self.device)

        self.g = MLP(2*self.embedding_dim,1) 

    def forward(self, nodes_u,nodes_v):#,  nodes_i,u_items_list, u_users_list,i_users_list,u_items_r,u_users_items,u_users_items_r,i_users_r):

        history_u_lists_batch = []
        history_ur_lists_batch = []
        history_v_lists_batch = []
        history_vr_lists_batch = []
        social_adj_lists_batch = []

        for u in nodes_u:
            history_u_lists_batch.append(self.history_u_lists[int(u)])
            history_ur_lists_batch.append(self.history_ur_lists[int(u)])

            social_adj_lists_batch.append(self.social_adj_lists[int(u)])

        for v in nodes_v:
            history_v_lists_batch.append(self.history_v_lists[int(v)])
            history_vr_lists_batch.append(self.history_vr_lists[int(v)])


        hi = self.model_user(nodes_u,history_u_lists_batch,history_ur_lists_batch,social_adj_lists_batch) #u_items_list, u_users_list,u_items_r,u_users_items,u_users_items_r,i_users_list,i_users_r)

        zj = self.model_item(nodes_v,history_v_lists_batch, self.history_vr_lists) #nodes_i,i_users_list,i_users_r)


        hi = F.relu(self.bn1(self.w_ur1(hi)))
        hi = F.dropout(hi, training=self.training)
        hi = self.w_ur2(hi)
        zj = F.relu(self.bn2(self.w_vr1(zj)))
        zj = F.dropout(zj, training=self.training)
        zj = self.w_vr2(zj)

        x = torch.cat((hi, zj), 1) #g
        x = F.relu(self.bn3(self.w_uv1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)




"""

#item-space latent factor
        pi = self.embed_u(nodes_u)
        # a in C(i): all items user has interacted with

        #select users items where ratings exist

        #print(u_items_r)

        er = self.embed_r(history_ur_lists_batch[:,:])

        qa = self.embed_i(history_u_lists_batch[:,:])

        xia = self.gv(torch.cat((qa,er),-1))

        #print(pi.shape,er.shape, qa.shape)#, xia.shape)
        
        alpha_ia = self.att_I(pi,xia)

        hi_I = F.relu(self.linI(torch.sum(alpha_ia*xia,1)))


        #social-space latent factor
        
        #compute ho_I for each user neighbours o
        
         #TO DO !!!!!!!!!!!!!!!!!
        
        po = self.embed_u(u_users_list)
        # a in C(i): all items user has interacted with

        #select users items where ratings exist

        ero = self.embed_r(u_users_items_r[:,:,:])

        qao = self.embed_i(u_users_items[:,:,:])

        xoa = self.gv(torch.cat((qao,ero),-1))

        #print(pi.shape,er.shape, qa.shape)#, xia.shape)
        
        alpha_oa = self.att_O(po,xoa)

        ho_I = F.relu(self.linO(torch.sum(alpha_oa*xoa,1)))

        ##social aggregation

        beta_io =  self.att_S(ho_I,pi)

        hi_S = F.relu(self.linO(torch.sum(beta_io*ho_I,1)))

        c1 =  torch.cat((hi_I,hi_S),-1)

        hi = self.mlp(c1)

"""