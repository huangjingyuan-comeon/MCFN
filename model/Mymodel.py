import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from dataloader import BasicDataset
from model.NeuralNetwork import *
from torch_geometric.nn import GATConv
# import dgl.function as fn
# import dgl.nn.pytorch as dglnn




class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()


    def getUsersRating(self, users):
        raise NotImplementedError

class ContrastLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self,featurex,featurey,label):
        device = (torch.device('cuda')
                  if featurex.is_cuda
                  else torch.device('cpu'))
        batch_size = featurex.shape[0]
        featurex = F.normalize(featurex,p=2,dim=1)
        featurey = F.normalize(featurey,p=2,dim=1)
        if type(label) == list:
            label = torch.tensor(label).to(device)
        labels = label.contiguous().view(1,-1)
        if labels.shape[1] != batch_size:
            return 0
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels,labels.T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(featurex, featurey.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # 构建mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class Muiti_PostGCN(BasicModel):
    def __init__(self,
                 config:dict,):
        super(Muiti_PostGCN,self).__init__()



        self.config =  config
        self.A_up = self.config['g_up']
        self.A_pu = self.A_up.T

        self.num_users = self.A_up.shape[0]
        self.num_posts = self.A_up.shape[1]
        self.embeding_size = self.config['embeding_size']
        self.keepprob = 0.6
        self.A_split = True
        self.n_layers = 2 # self.config['UPGCN_n_layers']
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.4)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embeding_size)
        self.embedding_post = torch.nn.Embedding(
            num_embeddings=self.num_posts, embedding_dim=self.embeding_size)


        self.f = nn.Sigmoid()
        self.__init__weight()

    def __init__weight(self):

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_post.weight, std=0.1)


    def computer(self):

        users_emb = self.embedding_user.weight
        posts_emb = self.embedding_post.weight
        users_emb_NR = self.embedding_user.weight
        posts_emb_NR = self.embedding_post.weight
        users_emb_FR = self.embedding_user.weight
        posts_emb_FR = self.embedding_post.weight
        users_emb_TR = self.embedding_user.weight
        posts_emb_TR = self.embedding_post.weight
        users_emb_UR = self.embedding_user.weight
        posts_emb_UR = self.embedding_post.weight


        X_post = [posts_emb]
        X_post_NR = [posts_emb_NR]
        X_post_FR = [posts_emb_FR]
        X_post_TR = [posts_emb_TR]
        X_post_UR = [posts_emb_UR]
        X_user = [users_emb]
        X_user_NR = [users_emb_NR]
        X_user_FR = [users_emb_FR]
        X_user_TR = [users_emb_TR]
        X_user_UR = [users_emb_UR]


        A_up  = self.A_up.todense()
        A_up = torch.FloatTensor(A_up).cuda()
        A_pu = A_up.T

        A_up_NR = self.config['A_up_NR'].todense()
        A_up_NR= torch.FloatTensor(A_up_NR).cuda()
        A_pu_NR = A_up_NR.T

        A_up_FR = self.config['A_up_FR'].todense()
        A_up_FR = torch.FloatTensor(A_up_FR).cuda()
        A_pu_FR = A_up_FR.T

        A_up_TR = self.config['A_up_TR'].todense()
        A_up_TR = torch.FloatTensor(A_up_TR).cuda()
        A_pu_TR = A_up_TR.T

        A_up_UR = self.config['A_up_UR'].todense()
        A_up_UR = torch.FloatTensor(A_up_UR).cuda()
        A_pu_UR = A_up_UR.T


        for i in range(self.n_layers):
            tempuser = users_emb
            temppost = posts_emb
            users_emb = torch.sparse.mm(A_up,temppost)
            posts_emb = torch.sparse.mm(A_pu,tempuser)
            X_post.append(posts_emb)
            X_user.append(users_emb)

            tempuser_NR = users_emb_NR
            temppost_NR = posts_emb_NR
            users_emb_NR = torch.sparse.mm(A_up_NR,temppost_NR)
            posts_emb_NR = torch.sparse.mm(A_pu_NR,tempuser_NR)
            X_post_NR.append(posts_emb_NR)
            X_user_NR.append(users_emb_NR)

            tempuser_FR = users_emb_FR
            temppost_FR = posts_emb_FR
            users_emb_FR = torch.sparse.mm(A_up_FR,temppost_FR)
            posts_emb_FR = torch.sparse.mm(A_pu_FR,tempuser_FR)
            X_post_FR.append(posts_emb_FR)
            X_user_FR.append(users_emb_FR)

            tempuser_TR = users_emb_TR
            temppost_TR = posts_emb_TR
            users_emb_TR = torch.sparse.mm(A_up_TR,temppost_TR)
            posts_emb_TR = torch.sparse.mm(A_pu_TR,tempuser_TR)
            X_post_TR.append(posts_emb_TR)
            X_user_TR.append(users_emb_TR)

            tempuser_UR = users_emb_UR
            temppost_UR = posts_emb_UR
            users_emb_UR = torch.sparse.mm(A_up_UR,temppost_UR)
            posts_emb_UR = torch.sparse.mm(A_pu_UR,tempuser_UR)
            X_post_UR.append(posts_emb_UR)
            X_user_UR.append(users_emb_UR)

        X_post = torch.stack(X_post,dim=1)
        X_post = torch.mean(X_post,dim=1)

        X_user = torch.stack(X_user, dim=1)
        X_user = torch.mean(X_user, dim=1)

        X_user_NR = torch.stack(X_user_NR, dim=1)
        X_user_NR = torch.mean(X_user_NR, dim=1)

        X_user_FR = torch.stack(X_user_FR, dim=1)
        X_user_FR = torch.mean(X_user_FR, dim=1)

        X_user_TR = torch.stack(X_user_TR, dim=1)
        X_user_TR = torch.mean(X_user_TR, dim=1)

        X_user_UR = torch.stack(X_user_UR, dim=1)
        X_user_UR = torch.mean(X_user_UR, dim=1)

        return  X_post,X_user,X_user_NR,X_user_FR,X_user_TR,X_user_UR



    def get_Userembedding(self, W, X_user, X_user_NR, X_user_FR, X_user_TR, X_user_UR):
        sim_NR = torch.cosine_similarity(X_user, X_user_NR, dim=1)
        sim_FR = torch.cosine_similarity(X_user, X_user_FR, dim=1)
        sim_TR = torch.cosine_similarity(X_user, X_user_TR, dim=1)
        sim_UR = torch.cosine_similarity(X_user, X_user_UR, dim=1)
        sim = torch.stack([sim_NR, sim_FR,sim_TR,sim_UR], dim=1)

        mask = (sim == sim.max(dim=1, keepdim=True)[0])
        sim = torch.mul(mask, sim)
        var = torch.std(sim,dim=1)
        var = var + 1
        var = torch.log(var)

        user = torch.stack([X_user_NR,X_user_FR,X_user_TR,X_user_UR],dim=1)
        user_embedding = torch.einsum('ab,abc->ac',W,user)
        A_up  = self.A_up.todense()
        A_up = torch.FloatTensor(A_up).cuda()
        A_pu = A_up.T
        #
        post_embedding = torch.sparse.mm(A_pu,user_embedding)
        #

        return user_embedding,post_embedding,var




    def forward(self,user_model,X_source_id,X_user_id,X_rumer_id,dropout = False,test = False,t=1):

        X_post,X_user, X_user_NR, X_user_FR, X_user_TR, X_user_UR = self.computer()
        X_post = X_post[X_source_id, :]
        mask = ((X_rumer_id != 0) == 0)
        AugUser,AugPost,var = self.get_Userembedding(user_model, X_user, X_user_NR, X_user_FR, X_user_TR, X_user_UR)


        X_var_user = X_user[X_rumer_id, :]
        X_var = var[X_rumer_id]
        X_var_user = torch.einsum('ab,abc->ac',X_var,X_var_user)


        AugUser = AugUser[X_rumer_id,:]
        AugUser = torch.mean(AugUser,dim=1)
        AugUser = torch.squeeze(AugUser,dim=1)

        X_mean_user = X_user[X_rumer_id, :]
        X_mean_user = torch.mean(X_mean_user,dim=1)
        X_mean_user = torch.squeeze(X_mean_user,dim=1)



        return X_post,X_var_user,AugUser,X_mean_user

class Muiti_PostGCN_weibo(BasicModel):
    def __init__(self,
                 config:dict):
        super(Muiti_PostGCN_weibo,self).__init__()
        # self.Graph = self.config['A_up']

        self.config =  config
        self.A_up = self.config['g_up']
        self.A_pu = self.A_up.T
        self.num_users = self.A_up.shape[0]
        self.num_posts = self.A_up.shape[1]
        self.embeding_size = self.config['embeding_size']
        self.keepprob = 0.6
        self.A_split = True
        self.n_layers = 2 # self.config['UPGCN_n_layers']
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.4)
        dropout_rate = config['dropout']
        embeding_size = self.config['embeding_size']
        alpha = 0.4
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embeding_size)
        self.embedding_post = torch.nn.Embedding(
            num_embeddings=self.num_posts, embedding_dim=self.embeding_size)

        # self.value = self.get_value()

        self.__init__weight()

    def __init__weight(self):

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_post.weight, std=0.1)


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        # index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, Graph,keep_prob):
        if self.A_split:
            graph = []
            for g in Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(Graph, keep_prob)
        return graph

    def computer(self):

        users_emb = self.embedding_user.weight
        posts_emb = self.embedding_post.weight
        users_emb_NR = self.embedding_user.weight
        posts_emb_NR = self.embedding_post.weight
        users_emb_FR = self.embedding_user.weight
        posts_emb_FR = self.embedding_post.weight

        X_post = [posts_emb]
        X_post_NR = [posts_emb_NR]
        X_post_FR = [posts_emb_FR]
        X_user = [users_emb]
        X_user_NR = [users_emb_NR]
        X_user_FR = [users_emb_FR]

        A_up = self.A_up.todense()
        A_up = torch.FloatTensor(A_up).cuda()
        A_pu = A_up.T

        A_up_NR = self.config['A_up_NR'].todense()
        A_up_NR = torch.FloatTensor(A_up_NR).cuda()
        A_pu_NR = A_up_NR.T

        A_up_FR = self.config['A_up_FR'].todense()
        A_up_FR = torch.FloatTensor(A_up_FR).cuda()
        A_pu_FR = A_up_FR.T

        for i in range(self.n_layers):
            tempuser = users_emb
            temppost = posts_emb
            users_emb = torch.sparse.mm(A_up, temppost)
            posts_emb = torch.sparse.mm(A_pu, tempuser)
            X_post.append(posts_emb)
            X_user.append(users_emb)

            tempuser_NR = users_emb_NR
            temppost_NR = posts_emb_NR
            users_emb_NR = torch.sparse.mm(A_up_NR, temppost_NR)
            posts_emb_NR = torch.sparse.mm(A_pu_NR, tempuser_NR)
            X_post_NR.append(posts_emb_NR)
            X_user_NR.append(users_emb_NR)

            tempuser_FR = users_emb_FR
            temppost_FR = posts_emb_FR
            users_emb_FR = torch.sparse.mm(A_up_FR, temppost_FR)
            posts_emb_FR = torch.sparse.mm(A_pu_FR, tempuser_FR)
            X_post_FR.append(posts_emb_FR)
            X_user_FR.append(users_emb_FR)


        X_post = torch.stack(X_post, dim=1)
        X_post = torch.mean(X_post, dim=1)

        X_user = torch.stack(X_user, dim=1)
        X_user = torch.mean(X_user, dim=1)

        X_user_NR = torch.stack(X_user_NR, dim=1)
        X_user_NR = torch.mean(X_user_NR, dim=1)

        X_user_FR = torch.stack(X_user_FR, dim=1)
        X_user_FR = torch.mean(X_user_FR, dim=1)



        return X_post, X_user, X_user_NR, X_user_FR

        return X_post, X_user, X_user0, X_user1

    def get_Userembedding(self, W,X_user, X_user_NR, X_user_FR):

        sim_NR = torch.cosine_similarity(X_user, X_user_NR, dim=1)
        sim_FR = torch.cosine_similarity(X_user, X_user_FR, dim=1)

        sim = torch.stack([sim_NR, sim_FR], dim=1)
        mask = (sim == sim.max(dim=1, keepdim=True)[0])
        sim = torch.mul(mask, sim)


        user = torch.stack([X_user_NR, X_user_FR], dim=1)


        user_embedding = torch.einsum('ab,abc->ac',W,user)

        var = torch.var(sim,dim=1)


        return user_embedding,var



    def forward(self ,W, X_source_id, X_user_id, X_rumer_id):

        X_post, X_user, X_user_NR, X_user_FR = self.computer()


        X_post = X_post[X_source_id, :]

        mask = ((X_rumer_id != 0) == 0)


        AugUser,var = self.get_Userembedding(W, X_user, X_user_NR, X_user_FR)

        X_var_user = X_user[X_rumer_id, :]

        X_var = var[X_rumer_id]
        X_var_user = torch.einsum('ab,abc->ac',X_var,X_var_user)

        AugUser = AugUser[X_rumer_id,:]
        AugUser = torch.mean(AugUser,dim=1)
        AugUser = torch.squeeze(AugUser,dim=1)

        X_mean_user = X_user[X_rumer_id, :]
        X_mean_user = torch.mean(X_mean_user,dim=1)
        X_mean_user = torch.squeeze(X_mean_user,dim=1)

        return X_post, X_mean_user, AugUser,X_var_user


class MCFN(NeuralNetwork):

    def __init__(self, config):
        super(MCFN, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        self.n_heads = config['n_heads']




        embeding_size = self.config['embeding_size']
        self.MultiPostGCN = Muiti_PostGCN(config)
        self.contraloss =ContrastLoss




        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))


        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=config['maxlen'] - K + 1) for K in config['kernel_sizes']])


        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.elu = nn.ELU()


        self.fc_out1 = nn.Sequential(
            nn.Linear( 300+3*embeding_size , 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, config["num_classes"])
        )

        self.init_weights()

    def init_weights(self):

        for name, param in self.fc_out1.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)


    def text_representation(self, X_word):
        X_word = X_word.permute(0, 2, 1)
        conv_block = []
        for Conv, max_pooling in zip(self.convs, self.max_poolings):
            act = self.relu(Conv(X_word))
            pool = max_pooling(act).squeeze()
            conv_block.append(pool)

        features = torch.cat(conv_block, dim= 1)
        features = self.dropout(features)
        return features

    def forward(self,user_model, X_source_wid, X_source_id, X_user_id, X_ruser_id,batch_y=None,Train = False):  # , X_composer_id, X_reviewer_id
        '''
        :param X_source_wid size: (batch_size, max_words)
                X_source_id size: (batch_size, )
                X_user_id  size: (batch_size, )
                X_retweet_id  size: (batch_size, max_retweets)
                X_retweet_uid  size: (batch_size, max_retweets)

        :return:
        '''

        X_word = self.word_embedding(X_source_wid)
        X_text = self.text_representation(X_word)


        source_rep,X_user, AugUser,AugPost = self.MultiPostGCN(user_model,X_source_id, X_user_id, X_ruser_id)
        if Train==True:
            con_loss = self.contraloss(source_rep,source_rep,batch_y)
        else:
            con_loss=0


        tweet_rep = torch.cat([X_text,source_rep,AugUser,AugPost], dim=1)


        Xt_logit = self.fc_out1(tweet_rep)



        return Xt_logit,con_loss


class MCFN_weibo(NeuralNetwork):

    def __init__(self, config):
        super(MCFN_weibo, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        self.n_heads = config['n_heads']



        embeding_size = self.config['embeding_size']

        self.MultiPostGCN = Muiti_PostGCN_weibo(config)
        self.contraloss =ContrastLoss


        dropout_rate = config['dropout']
        alpha = 0.4

        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))



        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=config['maxlen'] - K + 1) for K in config['kernel_sizes']])



        self.scale = torch.sqrt(torch.FloatTensor([embeding_size])).cuda() #  // self.n_heads



        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.contrast_loss = ContrastLoss()

        self.fc_out1 = nn.Sequential(
            nn.Linear( 300 + 3*embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, config["num_classes"])
        )


        self.init_weights()

    def init_weights(self):

        for name, param in self.fc_out1.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)


    def text_representation(self, X_word):
        X_word = X_word.permute(0, 2, 1)
        conv_block = []
        for Conv, max_pooling in zip(self.convs, self.max_poolings):
            act = self.relu(Conv(X_word))
            pool = max_pooling(act).squeeze()
            conv_block.append(pool)

        features = torch.cat(conv_block, dim=1)
        features = self.dropout(features)
        return features


    def ruser_representation(self, X_rumer):
        X_rumer = X_rumer.permute(0, 2, 1)
        conv_block = []
        for Conv, max_pooling in zip(self.rconvs, self.rmax_poolings):
            act = self.relu(Conv(X_rumer))
            pool = max_pooling(act).squeeze()
            conv_block.append(pool)

        features = torch.stack(conv_block, dim=1)
        features = torch.mean(features,dim=1)
        features = self.dropout(features)
        return features

    def forward(self,user_model, X_source_wid, X_source_id, X_user_id, X_ruser_id,batch_y=None,Train = False):  # , X_composer_id, X_reviewer_id

        '''
        :param X_source_wid size: (batch_size, max_words)
                X_source_id size: (batch_size, )
                X_user_id  size: (batch_size, )
                X_retweet_id  size: (batch_size, max_retweets)
                X_retweet_uid  size: (batch_size, max_retweets)

        :return:
        '''

        X_word = self.word_embedding(X_source_wid)
        X_text = self.text_representation(X_word)

        source_rep,X_user, AugUser,AugPost= self.MultiPostGCN(user_model,X_source_id, X_user_id, X_ruser_id)
        if Train == True:
            con_loss = self.contraloss(source_rep, source_rep, batch_y)
        else:
            con_loss = 0

        tweet_rep = torch.cat([X_text,source_rep,X_user,AugUser], dim=1)


        Xt_logit = self.fc_out1(tweet_rep)



        return Xt_logit,con_loss


