import numpy as np
import torch
from tqdm import tqdm

from .strategy import Strategy


class AdversarialDeepFool(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
                 X_img=None, X_txt=None, X_img_val=None, X_txt_val=None):
        super(AdversarialDeepFool, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
                                                  X_img, X_txt, X_img_val, X_txt_val)

    def cal_dis(self, x_img, x_txt):
        nx_txt = torch.unsqueeze(x_txt, 0)
        nx = torch.unsqueeze(x_img, 0)
        nx_txt.requires_grad_()
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).cuda()
        eta_txt = torch.zeros(nx_txt.shape).cuda()


        self.model.clf.cuda()
        common_feature_image, common_feature_text, \
        image_classifier_logits, text_classifier_logits, \
        label_feature, A, \
        real_image_out, fake_image_out, \
        fake_text_out, real_text_out = self.model.clf(nx+eta, nx_txt+eta_txt)
        n_class = image_classifier_logits.shape[1]
        py = image_classifier_logits.max(1)[1].item()
        ny = image_classifier_logits.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.args.max_iter:
            image_classifier_logits[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                image_classifier_logits[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = image_classifier_logits[0, i] - image_classifier_logits[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.cpu().numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.cpu().numpy().flatten()) * wi
            if ri is None:
                break
            eta += ri.clone()
            nx.grad.data.zero_()
            common_feature_image, common_feature_text, \
            image_classifier_logits, text_classifier_logits, \
            label_feature, A, \
            real_image_out, fake_image_out, \
            fake_text_out, real_text_out = self.model.clf(nx + eta, nx_txt + eta_txt)
            py = image_classifier_logits.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()


    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        self.model.clf.cpu()
        self.model.clf.eval()
        dis = np.zeros(idxs_unlabeled.shape)

        data_pool = self.model.handler(self.X_img[idxs_unlabeled], self.X_txt[idxs_unlabeled], self.Y[idxs_unlabeled])
        for i in tqdm(range(len(idxs_unlabeled))):
            x_img, x_txt, y, idx = data_pool[i]
            x_img, x_txt, y, idx = torch.Tensor(x_img).cuda(), torch.Tensor(x_txt).cuda(), torch.Tensor(y).cuda(), torch.Tensor(idx).cuda()
            dis[i] = self.cal_dis(x_img, x_txt)

        self.model.clf.cuda()

        probs, embeddings = self.predict_prob_embed(self.X_img[idxs_unlabeled], self.X_txt[idxs_unlabeled], self.Y[idxs_unlabeled])

        selected = dis.argsort()[:n]
        return idxs_unlabeled[selected], embeddings, probs, probs, selected, None


