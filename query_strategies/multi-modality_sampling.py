from torch import optim

from .strategy import Strategy
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision import models

class MultiModalSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
                 X_img = None, X_txt = None, X_img_val = None, X_txt_val = None):
        super(MultiModalSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
                                             X_img, X_txt, X_img_val, X_txt_val)



    def query(self, n, sample=True):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        img_embeds, txt_embeds = self.get_embedding(self.X_img, self.X_txt, self.Y, all_modal=True)

        img_dim = img_embeds.shape[1]
        txt_dim = txt_embeds.shape[1]
        unlb_img_embeds, unlb_txt_embeds = img_embeds[idxs_unlabeled], txt_embeds[idxs_unlabeled]
        lb_img_embeds, lb_txt_embeds = img_embeds[self.idxs_lb], txt_embeds[self.idxs_lb]

        features_img = torch.cat([unlb_img_embeds, lb_img_embeds], dim=0)
        features_txt = torch.cat([unlb_txt_embeds, lb_txt_embeds], dim=0)

        nlbl = np.arange(0, unlb_img_embeds.size(0), 1)
        lbl = np.arange(unlb_img_embeds.size(0), features_img.size(0), 1)

        bs = 128
        dataset = CustomDataSet(features_txt, features_img)
        dataloader = DataLoader(dataset, batch_size=bs, drop_last=False, shuffle=False, num_workers=12)
        model = CrossModalProbabilisticEncoder(n_head=1,
                                               d_in_img=img_dim, d_out_img=img_dim, d_h_img=img_dim // 2,
                                               dropout_img=0.2,
                                               d_in_txt=txt_dim, d_out_txt=txt_dim, d_h_txt=txt_dim // 2,
                                               dropout_txt=0.2, num_embeddings=self.args.K)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999), weight_decay=1e-4)
        criterion = Criterion()
        criterion.cuda()
        epochs = 40
        early_stop_times = 3
        lowest_loss = 1e20
        times = 0
        print('Learning Probabilistic Embedding Network...')
        model.train()
        for _ in tqdm(range(epochs)):
            model.train()
            running_loss = 0.0
            for index, (image, text) in enumerate(dataloader):
                optimizer.zero_grad()
                image = image.float().cuda()
                text = text.float().cuda()
                res = model(image, text)
                loss, loss_dict = criterion(**res)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    break
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            if running_loss < lowest_loss:
                lowest_loss = running_loss
                times = 0
            else:
                times += 1
            print('training_loss :{}'.format(running_loss/len(dataloader.dataset)))
            if times == early_stop_times:
                print('training is early stop!')
                break


        num_images = img_embeds.shape[0]
        num_texts = txt_embeds.shape[0]
        num_embeddings = model.num_embeddings
        feat_dim = img_dim
        model.eval()
        criterion.eval()
        image_features = torch.zeros((num_images, num_embeddings, feat_dim))
        text_features = torch.zeros((num_texts, num_embeddings, feat_dim))
        image_var = torch.zeros((num_images, feat_dim))
        text_var = torch.zeros((num_texts, feat_dim))
        with torch.no_grad():
            # extract feature
            for index, (image, text) in enumerate(dataloader):
                image = image.float().cuda()
                text = text.float().cuda()
                if sample:
                    res = model(image, text, sample=False)
                else:
                    res = model(image, text)
                _image_features = res['image_features']
                _text_features = res['caption_features']
                _image_var = res['image_logsigma']
                _text_var = res['caption_logsigma']
                image_features[index:index + _image_features.shape[0]] = _image_features
                text_features[index:index + _text_features.shape[0]] = _text_features
                image_var[index: index + _image_features.shape[0]] = _image_var
                text_var[index: index + _text_features.shape[0]] = _text_var

            scores = criterion.match_prob(image_features, text_features, image_var, text_var)
            selected = scores.sort()[1][:n].detach().cpu().numpy()
        del model, image_features, text_features, image_var, text_var
        torch.cuda.empty_cache()

        return idxs_unlabeled[selected], None, None, None, selected, None


class MultiLayerSelfAttention(nn.Module):
    def __init__(self, d_in, n_head, num_layers=4):
        super(MultiLayerSelfAttention, self).__init__()
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=d_in, num_heads=n_head, dropout=0.1) for _ in range(num_layers)])
    
    def forward(self, x):
        # Assuming x has shape (seq_len, batch_size, d_in)
        for attn_layer in self.attn_layers:
            # Apply attention to input, output will be fed to the next layer
            x, _ = attn_layer(x, x, x)
        return x

class ImageProbabilisiticEncoder(nn.Module):
    def __init__(self, n_head, d_in, d_out, dropout=0.0):
        super(ImageProbabilisiticEncoder, self).__init__()

        self.n_head = n_head
        self.self_attention = MultiLayerSelfAttention(d_in=d_in, n_head=n_head, num_layers=4)
        self.fc_for_mean = nn.Linear(d_in, d_out)
        self.fc_for_variance = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_for_mean.weight)
        nn.init.constant_(self.fc_for_mean.bias, 0)
        nn.init.xavier_uniform_(self.fc_for_variance.weight)
        nn.init.constant_(self.fc_for_variance.bias, 0)

    def forward(self, image, mask=None):
        # For the mean (using Sigmoid activation)
        residual_mean = self.self_attention(image)
        residual_mean = self.dropout(self.sigmoid(self.fc_for_mean(residual_mean)))
        mean = self.layer_norm(residual_mean + image)  # residual connection + LayerNorm

        # For the variance (using ReLU activation)
        residual_variance = self.self_attention(image)
        residual_variance = self.fc_for_variance(residual_variance)
        variance = self.relu(residual_variance + image)  # residual connection + ReLU activation
        
        return mean, variance

class TextProbabilisiticEncoder(nn.Module):
    def __init__(self, n_head, d_in, d_out, d_final, dropout=0.0):
        super(TextProbabilisiticEncoder, self).__init__()

        self.n_head = n_head
        self.self_attention = MultiLayerSelfAttention(d_in=d_in, n_head=n_head, num_layers=4)
        self.fc_for_mean = nn.Linear(d_in, d_out)
        self.fc_for_variance = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_final)
        self.fc1_final = nn.Linear(d_out, d_final)
        self.fc2_final = nn.Linear(d_out, d_final)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_for_mean.weight)
        nn.init.constant_(self.fc_for_mean.bias, 0)
        nn.init.xavier_uniform_(self.fc_for_variance.weight)
        nn.init.constant_(self.fc_for_variance.bias, 0)

    def forward(self, text, mask=None):
        # For the mean (using Sigmoid activation)
        residual_mean = self.self_attention(text)
        residual_mean = self.dropout(self.sigmoid(self.fc_for_mean(residual_mean)))
        mean = self.layer_norm(self.fc1_final(residual_mean + text))  # residual connection + LayerNorm

        # For the variance (using ReLU activation)
        residual_variance = self.self_attention(text)
        residual_variance = self.fc_for_variance(residual_variance)
        variance = self.fc2_final(self.relu(residual_variance + text))  # residual connection + ReLU activation
        
        return mean, variance



class CrossModalProbabilisticEncoder(nn.Module):
    def __init__(self, n_head, d_in_txt, d_out_txt, d_h_txt, dropout_txt,
                 d_in_img, d_out_img, d_h_img, dropout_img, num_embeddings):
        super(CrossModalProbabilisticEncoder, self).__init__()
        self.img_encoder = ImageProbabilisiticEncoder(n_head, d_in_img, d_out_img, d_h_img, dropout_img)
        self.txt_encoder = TextProbabilisiticEncoder(n_head, d_in_txt, d_out_txt, d_h_txt, d_out_img, dropout_txt)
        self.num_embeddings = num_embeddings

    def forward(self, image, text, sample=True):
        output = {}
        mean_img, var_img = self.img_encoder(image)
        mean_txt, var_txt = self.txt_encoder(text)
        if sample:
            output['image_features'] = sample_gaussian_tensors(mean_img, var_img, self.num_embeddings)
            output['caption_features'] = sample_gaussian_tensors(mean_txt, var_txt, self.num_embeddings)
        else:
            output['image_features'] = mean_img
            output['caption_features'] = mean_txt
        output['image_logsigma'] = var_img
        output['caption_logsigma'] = var_txt
        return output

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        shift = 15 * torch.ones(1)
        negative_scale = 15 * torch.ones(1)
        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)
        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

    def wasserstein_distance(self, mu_img, var_img, mu_txt, var_txt):
        return torch.nn.functional.normalize(mu_img - mu_txt, p=2) + \
               torch.nn.functional.normalize(var_txt - var_img, p=2)

    def var_dispersion(self, variance):
        return F.normalize(variance, p=2)

    def _compute_loss(self, input1, input2):
        distance, matched = self.pairwise_sampling(input1, input2)
        logits = -self.negative_scale * distance + self.shift

        idx = matched == 1
        loss_pos = self.soft_contrastive_nll(logits[idx], matched[idx]).sum()
        idx = matched != 1
        loss_neg = self.soft_contrastive_nll(logits[idx], matched[idx]).sum()

        return {
            'loss': loss_pos + loss_neg,
            'pos_loss': loss_pos,
            'neg_loss': loss_neg,
        }

    def pairwise_sampling(self, anchors, candidates):
        N = len(anchors)
        if len(anchors) != len(candidates):
            raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
        anchor_idx, selected_idx, matched = self.full_sampling(N)

        anchor_idx = torch.from_numpy(np.array(anchor_idx)).long()
        selected_idx = torch.from_numpy(np.array(selected_idx)).long()
        matched = torch.from_numpy(np.array(matched)).float()

        anchor_idx = anchor_idx.to(anchors.device)
        selected_idx = selected_idx.to(anchors.device)
        matched = matched.to(anchors.device)

        anchors = anchors[anchor_idx]
        selected = candidates[selected_idx]

        cdist = batchwise_cdist(anchors, selected)

        return cdist, matched

    def full_sampling(self, N):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i)
                selected.append(j)
                if i == j:
                    matched.append(1)
                else:
                    matched.append(-1)
        return candidates, selected, matched

    def soft_contrastive_nll(self, logit, matched):
        if len(matched.size()) == 1:
            matched = matched[:, None]
        return -(
            (logit * matched - torch.stack(
                (logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)
             ).logsumexp(dim=1)) + np.log(logit.size(1))

    def forward(self, mean_img, mean_txt, var_img, var_txt):
        w2_loss = self.wasserstein_distance(mean_img, var_img, mean_txt, var_txt)

        sampled_image_features, sampled_caption_features = mean_img, mean_txt


        i2t_loss = self._compute_loss(sampled_image_features, sampled_caption_features)
        t2i_loss = self._compute_loss(sampled_caption_features, sampled_image_features)
        loss = i2t_loss['loss'] + t2i_loss['loss'] + w2_loss

        loss_dict = {'i2t_loss': i2t_loss['loss'].item(),
                     't2i_loss': t2i_loss['loss'].item(),
                     'i2t_pos_loss': i2t_loss['pos_loss'].item(),
                     'i2t_neg_loss': i2t_loss['neg_loss'].item(),
                     't2i_pos_loss': t2i_loss['pos_loss'].item(),
                     't2i_neg_loss': t2i_loss['neg_loss'].item(),
                     'shift': self.shift.item(),
                     'negative_scale': self.negative_scale.item(),
                     'loss': loss.item()}
        return loss, loss_dict

    def match_prob(self, image_features, caption_features, image_var, caption_var):
        sampled_image_features, sampled_caption_features = image_features, caption_features
        distance = batchwise_cdist(sampled_image_features, sampled_caption_features)

        distance = distance.to(self.negative_scale.device)
        distance = distance.float()
        logits = -self.negative_scale * distance + self.shift
        prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))

        return prob.mean(axis=1) + self.var_dispersion(image_var) + self.var_dispersion(caption_var)

class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])
        self.vgg.classifier = MyClassifier()

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        # features = self.vgg_features(x).view(x.shape[0], -1)
        # features = self.fc_features(features)
        features = self.vgg.avgpool(self.vgg_features(x))
        return features

def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
        mu.unsqueeze(1))
    return samples

def batchwise_cdist(samples1, samples2, eps=1e-6):
    if len(samples1.size()) != 3 or len(samples2.size()) != 3:
        raise RuntimeError('expected: 3-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')

    samples1 = samples1.unsqueeze(1)
    samples2 = samples2.unsqueeze(2)
    return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)

class CustomDataSet(Dataset):
    def __init__(self, texts, images):
        self.texts = texts
        self.images = images
        self.im_div = 1
        if len(self.texts) != len(self.images):
            self.im_div = 5

    def __getitem__(self, idx):
        img_idx = idx // self.im_div
        return self.images[img_idx], self.texts[idx]

    def __len__(self):
        return len(self.texts)
