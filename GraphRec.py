import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


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



class UserModeling(nn.Module):
    def __init__(self,device,embed_u, embed_i, embed_r, embed_dim):
        super(UserModeling, self).__init__()

        self.device = device
        self.embed_u = embed_u
        self.embed_i = embed_i
        self.embed_r = embed_r
        self.embed_dim = embed_dim

        self.gv = MLP(2*self.embed_dim,self.embed_dim)
        self.att_I = Attention(2*self.embed_dim)
        self.att_O = Attention(2*self.embed_dim)
        self.att_S = Attention(2*self.embed_dim)

        self.linI = nn.Linear(self.embed_dim,self.embed_dim)
        self.linO = nn.Linear(self.embed_dim,self.embed_dim)
        self.linS = nn.Linear(self.embed_dim,self.embed_dim)

        self.mlp = MLP(2*self.embed_dim,self.embed_dim)
        

    def forward(self,u_idx,u_items_list, u_users_list,u_items_r,u_users_items,u_users_items_r):

        #item-space latent factor
        pi = self.embed_u(u_idx)
        # a in C(i): all items user has interacted with

        #select users items where ratings exist

        er = self.embed_r(u_items_r[:,:])

        qa = self.embed_i(u_items_list[:,:])

        xia = self.gv(torch.cat((qa,er),-1))

        #print(pi.shape,er.shape, qa.shape)#, xia.shape)
        
        alpha_ia = self.att_I(pi,xia)

        hi_I = F.relu(self.linI(torch.sum(alpha_ia*xia,1)))


        #social-space latent factor
        
        #compute ho_I for each user neighbours o
        """ 
         TO DO !!!!!!!!!!!!!!!!!
        """
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

        return hi


class ItemModeling(nn.Module):
    def __init__(self,device,embed_i, embed_u, embed_r, embed_dim):
        super(ItemModeling, self).__init__()

        self.device = device
        self.embed_u = embed_u
        self.embed_i = embed_i
        self.embed_r = embed_r
        self.embed_dim = embed_dim

        self.gu = MLP(2*self.embed_dim,self.embed_dim)
        self.att = Attention(2*self.embed_dim)

        self.lin = nn.Linear(self.embed_dim,self.embed_dim)
        

    def forward(self,i_idx,i_users_list,i_users_r):

        #user-aggregation
        qj = self.embed_i(i_idx)

        pt = self.embed_u(i_users_list[:,:])

        er = self.embed_r(i_users_r[:,:])

        #print(qj.shape,er.shape,pt.shape)

        fjt = self.gu(torch.cat((pt,er),-1))

        mu_jt = self.att(qj,fjt)

        zj = F.relu(self.lin(torch.sum(mu_jt*fjt,1)))

        return zj



class GraphRec(nn.Module):
    def __init__(self, device,num_users, num_items, num_ratings_values, embedding_dim = 64):
        super(GraphRec, self).__init__()

        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings_values = num_ratings_values
        self.embedding_dim = embedding_dim

        self.embed_u = nn.Embedding(self.num_users,self.embedding_dim) #p
        self.embed_i = nn.Embedding(self.num_items,self.embedding_dim) #q
        self.embed_r = nn.Embedding(self.num_ratings_values,self.embedding_dim) #er

        self.model_user = UserModeling(self.device,self.embed_u, self.embed_i, self.embed_r,self.embedding_dim)
        self.model_item = ItemModeling(self.device,self.embed_i, self.embed_u, self.embed_r,self.embedding_dim)

        self.g = MLP(2*self.embedding_dim,1) 

    def forward(self, nodes_u,nodes_i,u_items_list, u_users_list,i_users_list,u_items_r,u_users_items,u_users_items_r,i_users_r):
        hi = self.model_user(nodes_u,u_items_list, u_users_list,u_items_r,u_users_items,u_users_items_r)
        zj = self.model_item(nodes_i,i_users_list,i_users_r)

        g1 = torch.cat((hi,zj),-1)

        gl = self.g(g1)

        return gl



