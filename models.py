import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal_

class Embedding(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(Embedding, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop_in = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.drop_1 = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.drop_2 = nn.Dropout(p=dropout)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.drop_in(self.norm(x))
        x = F.relu(self.linear_1(x)); x = self.drop_1(x)
        x = F.relu(self.linear_2(x)); x = self.drop_2(x)
        x = F.relu(self.linear_3(x))
        return x

class TwoFeatureLMF(nn.Module):
    def __init__(self, input_dims, hidden_dims, dropout, output_dim, rank, use_softmax=False):
        super(TwoFeatureLMF, self).__init__()
        self.X1_in = input_dims[0]
        self.X2_in = input_dims[1]
        self.X1_hidden = hidden_dims[0]
        self.X2_hidden = hidden_dims[1]
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax
        self.X1_prob = dropout[0]
        self.X2_prob = dropout[1]
        self.post_fusion_prob = dropout[2]

        self.X1_subnet = Embedding(self.X1_in, self.X1_hidden, self.X1_prob)
        self.X2_subnet = Embedding(self.X2_in, self.X2_hidden, self.X2_prob)
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

        self.X1_factor = nn.Parameter(torch.Tensor(self.rank, self.X1_hidden + 1, self.output_dim))
        self.X2_factor = nn.Parameter(torch.Tensor(self.rank, self.X2_hidden + 1, self.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        xavier_normal_(self.X1_factor)
        xavier_normal_(self.X2_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, X1, X2):
        X1_h = self.X1_subnet(X1)
        X2_h = self.X2_subnet(X2)
        batch_size = X1_h.shape[0]
        DTYPE = torch.cuda.FloatTensor if X1_h.is_cuda else torch.FloatTensor

        _X1_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), X1_h), dim=1)
        _X2_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), X2_h), dim=1)

        fusion_X1 = torch.matmul(_X1_h, self.X1_factor)
        fusion_X2 = torch.matmul(_X2_h, self.X2_factor)
        fusion_zy = fusion_X1 * fusion_X2
        transposed_feature = fusion_zy[:, :, 0].transpose(0, 1)
        return transposed_feature

# 三特征 LMF 类
class ThreeFeatureLMF(nn.Module):
    def __init__(self, input_dims, hidden_dims, dropout, output_dim, rank, use_softmax=False):
        super(ThreeFeatureLMF, self).__init__()
        self.X1_in = input_dims[0]
        self.X2_in = input_dims[1]
        self.X3_in = input_dims[2]
        self.X1_hidden = hidden_dims[0]
        self.X2_hidden = hidden_dims[1]
        self.X3_hidden = hidden_dims[2]
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.X1_prob = dropout[0]
        self.X2_prob = dropout[1]
        self.X3_prob = dropout[2]
        self.post_fusion_prob = dropout[3]

        self.X1_subnet = Embedding(self.X1_in, self.X1_hidden, self.X1_prob)
        self.X2_subnet = Embedding(self.X2_in, self.X2_hidden, self.X2_prob)
        self.X3_subnet = Embedding(self.X3_in, self.X3_hidden, self.X3_prob)
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

        self.X1_factor = nn.Parameter(torch.Tensor(self.rank, self.X1_hidden + 1, self.output_dim))
        self.X2_factor = nn.Parameter(torch.Tensor(self.rank, self.X2_hidden + 1, self.output_dim))
        self.X3_factor = nn.Parameter(torch.Tensor(self.rank, self.X3_hidden + 1, self.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        xavier_normal_(self.X1_factor)
        xavier_normal_(self.X2_factor)
        xavier_normal_(self.X3_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, X1, X2, X3):
        X1_h = self.X1_subnet(X1)
        X2_h = self.X2_subnet(X2)
        X3_h = self.X3_subnet(X3)
        batch_size = X1_h.shape[0]
        DTYPE = torch.cuda.FloatTensor if X1.is_cuda else torch.FloatTensor

        _X1_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), X1_h), dim=1)
        _X2_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), X2_h), dim=1)
        _X3_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), X3_h), dim=1)

        fusion_X1 = torch.matmul(_X1_h, self.X1_factor)
        fusion_X2 = torch.matmul(_X2_h, self.X2_factor)
        fusion_X3 = torch.matmul(_X3_h, self.X3_factor)
        fusion_zy = fusion_X1 * fusion_X2 * fusion_X3

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output

# 整体模型类
class AllFeatureModel(nn.Module):
    def __init__(self, model_feat12, model_feat23, model_feat13, three_featurn_LMF):
        super(AllFeatureModel, self).__init__()
        self.model_feat12 = model_feat12
        self.model_feat23 = model_feat23
        self.model_feat13 = model_feat13
        self.three_featurn_LMF = three_featurn_LMF

    def forward(self, X1, X2, X3):
        output_feat12 = self.model_feat12(X1, X2)
        output_feat23 = self.model_feat23(X2, X3)
        output_feat13 = self.model_feat13(X1, X3)
        output = self.three_featurn_LMF(output_feat12, output_feat23, output_feat13)
        return output
