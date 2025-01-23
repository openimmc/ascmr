import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from torchvision import models
# from torch_geometric.nn import GINConv

from util import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2)
        self.CGCN = GCN(nfeat, nhid1, nhid2)
        self.attention = Attention(nhid2)

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN1(x, sadj)  # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj)  # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        # attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        return emb, att, emb1, com1, com2, emb2


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class TxtMLP(nn.Module):
    def __init__(self, code_len=300, txt_bow_len=1386, num_class=24):
        super(TxtMLP, self).__init__()
        self.fc1 = nn.Linear(txt_bow_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.classifier = nn.Linear(code_len, num_class)

    def forward(self, x):
        feat = F.leaky_relu(self.fc1(x), 0.2)
        feat = F.leaky_relu(self.fc2(feat), 0.2)
        predict = self.classifier(feat)
        return feat, predict


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class IDCM_NN(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,
                 adj_file=None, inp=None):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.classifier = SFGCN(in_channel, 512, minus_one_dim)

        self.f_adj = Parameter(gen_fea(inp), requires_grad=False)
        self.s_adj = Parameter(gen_adj(torch.FloatTensor(gen_A(num_classes, t, adj_file))))
        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)
        view1_feature = F.normalize(view1_feature, dim=1)
        view2_feature = F.normalize(view2_feature, dim=1)

        x, _, emb1, com1, com2, emb2 = self.classifier(self.inp, self.s_adj, self.f_adj)
        x = F.normalize(x, dim=1)
        y = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, y)
        y_text = torch.matmul(view2_feature, y)
        return view1_feature, view2_feature, y_img, y_text, x, emb1, com1, com2, emb2


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class ModalClassifier(nn.Module):
    """Network to discriminate modalities"""

    def __init__(self, input_dim=40):
        super(ModalClassifier, self).__init__()
        self.denseL1 = nn.Linear(input_dim, input_dim // 4)
        self.denseL2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.denseL3 = nn.Linear(input_dim // 16, 2)

    def forward(self, x):
        x = ReverseLayerF.apply(x, 1.0)
        out = self.denseL1(x)
        out = self.denseL2(out)
        out = self.denseL3(out)
        return out


class ImgDec(nn.Module):
    """Network to decode image representations"""

    def __init__(self, input_dim=1024, output_dim=4096, hidden_dim=2048):
        super(ImgDec, self).__init__()
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        out = F.relu(self.denseL2(out))
        return out


class TextDec(nn.Module):
    """Network to decode image representations"""

    def __init__(self, input_dim=1024, output_dim=300, hidden_dim=512):
        super(TextDec, self).__init__()
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.leaky_relu(self.denseL1(x), 0.2)
        out = F.leaky_relu(self.denseL2(out), 0.2)
        return out


# class ImgDec(nn.Module):
#     """Network to decode image representations"""
#
#     def __init__(self, input_dim=1024, output_dim=4096, hidden_dim=2048):
#         super(ImgDec, self).__init__()
#         self.denseL1 = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         out = F.relu(self.denseL1(x))
#         # out = F.relu(self.denseL2(out))
#         return out
#
#
# class TextDec(nn.Module):
#     """Network to decode image representations"""
#
#     def __init__(self, input_dim=1024, output_dim=300, hidden_dim=512):
#         super(TextDec, self).__init__()
#         self.denseL1 = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         out = F.leaky_relu(self.denseL1(x), 0.2)
#         # out = F.leaky_relu(self.denseL2(out), 0.2)
#         return out

class D_SE_I(nn.Module):
    pass


class D_SE_T(nn.Module):
    pass


class DALGNN2(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,
                 adj_file=None, inp=None):
        super(DALGNN2, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.img_md_net = ModalClassifier(img_input_dim)
        self.text_md_net = ModalClassifier(text_input_dim)
        self.classifier = SFGCN(in_channel, 512, minus_one_dim)

        self.f_adj = Parameter(gen_fea(inp), requires_grad=False)
        self.s_adj = Parameter(gen_adj(torch.FloatTensor(gen_A(num_classes, t, adj_file))))

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)
        # view1_feature = F.normalize(view1_feature, dim=1)
        # view2_feature = F.normalize(view2_feature, dim=1)

        x, _, emb1, com1, com2, emb2 = self.classifier(self.inp, self.s_adj, self.f_adj)
        x = F.normalize(x, dim=1)
        y = x.transpose(0, 1)

        y_img = torch.matmul(F.normalize(view1_feature, dim=1), y)
        y_text = torch.matmul(F.normalize(view2_feature, dim=1), y)

        view1_feature_view2 = self.img2text_net(view1_feature)
        view2_feature_view1 = self.text2img_net(view2_feature)
        view1_modal_view1 = self.img_md_net(feature_img)
        view2_modal_view1 = self.img_md_net(view2_feature_view1)
        view1_modal_view2 = self.text_md_net(view1_feature_view2)
        view2_modal_view2 = self.text_md_net(feature_text)

        return view1_feature, view2_feature, y_img, y_text, x, \
               view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2, \
               emb1, com1, com2, emb2


class DALGNN(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,
                 adj_file=None, inp=None, GNN='GAT', n_layers=4):
        super(DALGNN, self).__init__()
        self.n_label = num_classes
        self.embedding_dim = minus_one_dim
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.img_md_net = ModalClassifier(img_input_dim)
        self.text_md_net = ModalClassifier(text_input_dim)
        self.num_classes = num_classes
        if GNN == 'GAT':
            self.gnn = GraphAttentionLayer
        elif GNN == 'GCN':
            self.gnn = GraphConvolution
        elif GNN == 'GIN':
            self.gnn = GINConv
        else:
            raise NameError("Invalid GNN name!")
        self.n_layers = n_layers

        self.relu = nn.LeakyReLU(0.2)
        if GNN == 'GIN':
            self.lrn = [self.gnn(nn.Sequential(nn.Linear(in_channel, minus_one_dim), nn.ReLU()))]
        else:
            self.lrn = [self.gnn(in_channel, minus_one_dim)]
        for i in range(1, self.n_layers):
            if GNN == 'GIN':
                self.lrn.append(self.gnn(nn.Sequential(nn.Linear(minus_one_dim, minus_one_dim), nn.ReLU())))
            else:
                self.lrn.append(self.gnn(minus_one_dim, minus_one_dim))
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)
        self.hypo = nn.Linear(self.n_layers * minus_one_dim, minus_one_dim)

        _adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))
        if GNN == 'GAT':
            self.adj = Parameter(_adj, requires_grad=False)
        else:
            # self.adj = Parameter(gen_adj(_adj), requires_grad=False)
            self.adj = Parameter(_adj, requires_grad=False)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        # 正交初始化试一下
        self.W1 = Parameter(torch.Tensor(in_channel, in_channel))
        stdv = 1. / math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stdv, stdv)
        self.W2 = Parameter(torch.Tensor(in_channel, in_channel))
        stdv = 1. / math.sqrt(self.W2.size(1))
        self.W2.data.uniform_(-stdv, stdv)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.t = 0.07
        self.encoder = nn.Sequential(nn.Linear(minus_one_dim, minus_one_dim), nn.ReLU(),
                                     nn.Linear(minus_one_dim, num_classes))

    def info(self, view1_feature, view2_feature, labels):
        view1_predict = F.normalize(self.encoder(view1_feature), dim=1)
        view2_predict = F.normalize(self.encoder(view2_feature), dim=1)
        # cosine similarity: NxN
        sim_view12 = torch.matmul(view1_predict, view2_predict.T) / self.t
        sim_view11 = torch.matmul(view1_predict, view1_predict.T) / self.t
        sim_view22 = torch.matmul(view2_predict, view2_predict.T) / self.t
        mask = torch.matmul(labels, labels.T).clamp(min=0, max=1.0)
        mask_intra = mask - torch.eye(mask.shape[0]).cuda()

        # logits: NxN
        logits_view12 = sim_view12 - torch.log(torch.exp(1.05 * sim_view12).sum(1, keepdim=True))
        logits_view21 = sim_view12.T - torch.log(torch.exp(1.05 * sim_view12.T).sum(1, keepdim=True))
        logits_view11 = sim_view11 - torch.log(torch.exp(1.05 * sim_view11).sum(1, keepdim=True))
        logits_view22 = sim_view22 - torch.log(torch.exp(1.05 * sim_view22).sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos_view12 = (mask * logits_view12).sum(1) / mask.sum(1).clamp(min=1e-6)
        mean_log_prob_pos_view21 = (mask * logits_view21).sum(1) / mask.sum(1).clamp(min=1e-6)
        mean_log_prob_pos_view11 = (mask_intra * logits_view11).sum(1) / mask_intra.sum(1).clamp(min=1e-6)
        mean_log_prob_pos_view22 = (mask_intra * logits_view22).sum(1) / mask_intra.sum(1).clamp(min=1e-6)

        # supervised cross-modal contrastive loss
        loss = - mean_log_prob_pos_view12.mean() - mean_log_prob_pos_view21.mean() \
               - 0.1 * (mean_log_prob_pos_view11.mean() + mean_log_prob_pos_view22.mean())

        return loss

    def forward(self, feature_img, feature_text):
        common_feature_image = self.img_net(feature_img)
        common_feature_text = self.text_net(feature_text)

        layers = []
        x = self.inp
        A = torch.matmul(x.mm(self.W1), x.mm(self.W2).T) / self.num_classes
        # adj = gen_adj(torch.relu(0.5 * self.adj + 0.5 * A))
        adj = gen_adj(self.adj)
        for i in range(self.n_layers):
            x = self.lrn[i](x, adj)
            if self.gnn == GraphConvolution:
                x = self.relu(x)
            layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)
        x = F.normalize(x, dim=1)

        x = x.transpose(0, 1)
        image_classifier_logits = torch.matmul(F.normalize(common_feature_image, dim=1), x)
        text_classifier_logits = torch.matmul(F.normalize(common_feature_text, dim=1), x)

        view1_feature_view2 = self.img2text_net(common_feature_image)
        view2_feature_view1 = self.text2img_net(common_feature_text)

        view1_modal_view1 = self.img_md_net(feature_img)
        view2_modal_view1 = self.img_md_net(view2_feature_view1)
        view1_modal_view2 = self.text_md_net(view1_feature_view2)
        view2_modal_view2 = self.text_md_net(feature_text)

        return common_feature_image, common_feature_text, \
               image_classifier_logits, text_classifier_logits, \
               x.transpose(0, 1), A, \
               view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_classifier(self):
        return self.fc2

class CosineClassifier(nn.Module):
    def __init__(self, hidden_dim, num_class, W):
        super(CosineClassifier, self).__init__()
        if W is None:
            W = torch.Tensor(hidden_dim, hidden_dim)
            W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: num_class]
        self.W = Parameter(W)

    def forward(self, x):
        classifier = F.normalize(self.W, dim=0)
        fea = F.normalize(x, dim=1)
        out = torch.matmul(fea, classifier)

        return out


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape:
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static  # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x


class ADD_GCN(nn.Module):
    def __init__(self, num_classes, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024):
        super(ADD_GCN, self).__init__()
        self.g1_img_net = ImgNN(img_input_dim, minus_one_dim)
        self.g1_text_net = TextNN(text_input_dim, minus_one_dim)
        self.g2_img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.g2_text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.image_feature_discriminator = ModalClassifier(img_input_dim)
        self.text_feature_discriminator = ModalClassifier(text_input_dim)
        self.num_classes = num_classes

        W = torch.Tensor(minus_one_dim, minus_one_dim)
        W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: num_classes]
        self.fc = CosineClassifier(minus_one_dim, num_classes, W)
        self.relu = nn.LeakyReLU(0.2)
        self.gcn = DynamicGraphConvolution(minus_one_dim, minus_one_dim, num_classes)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = Parameter(W)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.t = 0.07
        self.encoder = nn.Sequential(nn.Linear(minus_one_dim, minus_one_dim), nn.ReLU(),
                                     nn.Linear(minus_one_dim, num_classes))

    def info(self, view1_feature, view2_feature, labels):
        view1_predict = F.normalize(self.encoder(view1_feature), dim=1)
        view2_predict = F.normalize(self.encoder(view2_feature), dim=1)
        # cosine similarity: NxN
        sim_view12 = torch.matmul(view1_predict, view2_predict.T) / self.t
        sim_view11 = torch.matmul(view1_predict, view1_predict.T) / self.t
        sim_view22 = torch.matmul(view2_predict, view2_predict.T) / self.t
        mask = torch.matmul(labels, labels.T).clamp(min=0, max=1.0)
        mask_intra = mask - torch.eye(mask.shape[0]).cuda()

        # logits: NxN
        logits_view12 = sim_view12 - torch.log(torch.exp(1.05 * sim_view12).sum(1, keepdim=True))
        logits_view21 = sim_view12.T - torch.log(torch.exp(1.05 * sim_view12.T).sum(1, keepdim=True))
        logits_view11 = sim_view11 - torch.log(torch.exp(1.05 * sim_view11).sum(1, keepdim=True))
        logits_view22 = sim_view22 - torch.log(torch.exp(1.05 * sim_view22).sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos_view12 = (mask * logits_view12).sum(1) / mask.sum(1).clamp(min=1e-6)
        mean_log_prob_pos_view21 = (mask * logits_view21).sum(1) / mask.sum(1).clamp(min=1e-6)
        mean_log_prob_pos_view11 = (mask_intra * logits_view11).sum(1) / mask_intra.sum(1).clamp(min=1e-6)
        mean_log_prob_pos_view22 = (mask_intra * logits_view22).sum(1) / mask_intra.sum(1).clamp(min=1e-6)

        # supervised cross-modal contrastive loss
        loss = - mean_log_prob_pos_view12.mean() - mean_log_prob_pos_view21.mean() \
               - 0.1 * (mean_log_prob_pos_view11.mean() + mean_log_prob_pos_view22.mean())

        return loss

    def forward_sam(self, x):
        """ SAM module

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    # def forward_dgcn(self, x):
    #     x = self.gcn(x)
    #     return x

    def forward(self, feature_img, feature_text):
        ### G1
        common_feature_image = self.g1_img_net(feature_img)
        common_feature_text = self.g1_text_net(feature_text)

        image_classifier_logits = self.fc(common_feature_image)
        text_classifier_logits = self.fc(common_feature_text)

        ###
        image_attention_weight = torch.sigmoid(image_classifier_logits)
        image_attention_weight = image_attention_weight.unsqueeze(1)

        text_attention_weight = torch.sigmoid(text_classifier_logits)
        text_attention_weight = text_attention_weight.unsqueeze(1)

        image_label_init_feature = torch.matmul(common_feature_image.unsqueeze(-1), image_attention_weight)
        text_label_init_feature = torch.matmul(common_feature_text.unsqueeze(-1), text_attention_weight)

        image_label_gcn_feature = self.gcn(image_label_init_feature)
        image_label_gcn_feature = image_label_init_feature + image_label_gcn_feature

        text_label_gcn_feature = self.gcn(text_label_init_feature)
        text_label_gcn_feature = text_label_init_feature + text_label_gcn_feature

        common_feature_classifier = F.normalize(self.last_linear, dim=0)
        final_image_classifier_logits = F.normalize(image_label_gcn_feature,
                                                    dim=1) * common_feature_classifier  # B*1*num_classes
        final_text_classifier_logits = F.normalize(text_label_gcn_feature,
                                                   dim=1) * common_feature_classifier  # B*1*num_classes
        # mask_mat = self.mask_mat.detach()
        final_image_classifier_logits = final_image_classifier_logits.sum(1)
        final_text_classifier_logits = final_text_classifier_logits.sum(1)

        ### G2
        text_feature_by_image = self.g2_img2text_net(common_feature_image)
        image_feature_by_text = self.g2_text2img_net(common_feature_text)

        ### D
        real_image_out = self.image_feature_discriminator(feature_img)
        fake_image_out = self.image_feature_discriminator(image_feature_by_text)

        real_text_out = self.text_feature_discriminator(feature_text)
        fake_text_out = self.text_feature_discriminator(text_feature_by_image)

        return common_feature_image, \
               common_feature_text, \
               image_classifier_logits, \
               text_classifier_logits, \
               final_image_classifier_logits, \
               final_text_classifier_logits, \
               real_image_out, \
               fake_image_out, \
               fake_text_out, \
               real_text_out
