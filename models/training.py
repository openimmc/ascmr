import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import scipy
import scipy.spatial
from sklearn.metrics import accuracy_score

from itr_models.evaluate import fx_calc_map_label
from itr_models.mmd import mix_rbf_mmd2


class Training(object):
    def __init__(self, net, net_args, handler, args, writer, device, init_model=True):
        self.net = net  # MLP
        self.net_args = net_args
        self.handler = handler  # dataset

        self.args = args
        self.writer = writer
        self.device = device

        if init_model:
            if self.args['task'] == 'ITR':
                self.clf = self.net.to(self.device)
            else:
                self.clf = self.net(**self.net_args).to(self.device)
            self.initial_state_dict = copy.deepcopy(self.clf.state_dict())

            print(self.clf)

    def _validate(self, X_val, Y_val, name, epoch):
        if X_val is None or len(X_val) <= 0:
            return

        P = self.predict(X_val, Y_val)
        acc = 1.0 * (Y_val == P).sum().item() / len(Y_val)

        self.writer.add_scalar('vaidation_accuracy/%s' % name, acc, epoch)
        #print('%s validation accuracy at epoch %d: %f' % (name, epoch, acc))

        return acc

    def _mixup_train(self, epoch, loader_tr, optimizer, name):
        self.clf.train()
        criterion = nn.CrossEntropyLoss()


        accFinal, tot_loss, iters = 0., 0., 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            if y.size(0) <= 1:
                continue
            optimizer.zero_grad()

            inputs, targets_a, targets_b, lam = self.mixup_data(x, y)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            out, e1 = self.clf(inputs)
            loss = self.mixup_criterion(criterion, out, targets_a, targets_b, lam)

            tot_loss += loss.item()
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()

            loss.backward()
            optimizer.step()

            self.iter += 1
            iters += 1

        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)
        return accFinal / len(loader_tr.dataset.X)

    def mixup_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        alpha = self.args['mixup_alpha']
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        if self.args['mixup_max_lambda']:
            lam = max(lam, 1 - lam)

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _mixup_hidden_train(self, epoch, loader_tr, optimizer, name):
        self.clf.train()
        criterion = nn.CrossEntropyLoss()

        # import ipdb; ipdb.set_trace()

        accFinal, tot_loss, iters = 0., 0., 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            if y.size(0) <= 1:
                continue
            optimizer.zero_grad()


            _, embedding = self.clf(x)

            inputs, targets_a, targets_b, lam = self.mixup_data(embedding, y)

            out, embedding = self.clf(inputs, embedding=True)

            loss = self.mixup_criterion(criterion, out, targets_a, targets_b, lam)

            tot_loss += loss.item()
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()

            loss.backward()
            optimizer.step()

            self.iter += 1
            iters += 1

        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)
        return accFinal / len(loader_tr.dataset.X)

    # train one epoch
    def _train(self, epoch, loader_tr, optimizer, name, scheduler=None):
        self.clf.train()
        dt_size = len(loader_tr)
        accFinal, tot_loss, iters = 0., 0., 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            if y.size(0) <= 1:
                continue
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)

            tot_loss += loss.item()
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()

            loss.backward()
            optimizer.step()


            self.iter += 1
            iters += 1

            if scheduler is not None:
                scheduler.step(epoch - self.args['lr_warmup'] + iters/float(dt_size))

        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)
        return accFinal / len(loader_tr.dataset.X)

    def train(self, name, X, Y, idxs_lb, X_val, Y_val, train_epoch_func=None):
        n_epoch = 2000 if self.args['n_epoch'] <= 0 else self.args['n_epoch']
        #n_val_iter = self.args['n_val_iter']
        if not self.args['continue_training']:
            self.clf = self.net(**self.net_args).to(self.device)
            self.clf.load_state_dict(copy.deepcopy(self.initial_state_dict))

        if self.args['optimizer'] == 'Adam':
            print('Adam optimizer...')
            optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])
        else:
            print('SGD optimizer...')
            optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        optimizer.zero_grad()

        if self.args['lr_schedule']:
            for param in optimizer.param_groups:
                param['initial_lr'] = param['lr']

            scheduler = CosineAnnealingLR(optimizer, n_epoch, eta_min=0)
        else:
            scheduler = None

        idxs_train = np.arange(len(Y))[idxs_lb]
        loader_tr = DataLoader(self.handler(X[idxs_train], Y[idxs_train], transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args']) # 仅用label pool的数据进行训练

        self.iter = 0
        self.best_model = None
        best_acc, best_epoch, n_stop = 0., 0, 0

        lr = self.args['optimizer_args']['lr']
        print('Training started...')
        for epoch in tqdm(range(n_epoch)):
            if self.args['lr_decay_epochs'] is not None and epoch in self.args['lr_decay_epochs']:
                for param in optimizer.param_groups:
                    lr /= 10.
                    param['lr'] = lr
                    param['initial_lr'] = lr

                    if self.args['lr_schedule']:
                        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.args['lr_T_0'], T_mult=self.args['lr_T_mult'])

            if epoch < self.args['lr_warmup']:
                learning_rate = lr * (epoch + 1) / float(self.args['lr_warmup'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            for param in optimizer.param_groups:
                self.writer.add_scalar('learning_rate/%s' % name, param['lr'], epoch)

            if train_epoch_func is not None:
                accCurrent = train_epoch_func(epoch, loader_tr, optimizer, name,
                                     scheduler=(None if epoch < self.args['lr_warmup'] else scheduler))
            else:
                accCurrent = self._train(epoch, loader_tr, optimizer, name,
                                         scheduler=(None if epoch < self.args['lr_warmup'] else scheduler))

            self.writer.add_scalar('training_accuracy/%s' % name, accCurrent, epoch)

            if X_val is not None and len(X_val) > 0:
                val_acc = self._validate(X_val, Y_val, name, epoch)
                if val_acc is not None:
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_epoch = epoch
                        n_stop = 0
                        if self.args['choose_best_val_model']:
                            self.best_model = copy.deepcopy(self.clf)
                    else:
                        n_stop += 1

                    if n_stop > self.args['n_early_stopping']:
                        print('Early stopping at epoch %d ' % epoch)
                        break

            if not self.args['train_to_end'] and accCurrent >= 0.99:
                print('Reached max accuracy at epoch %d ' % epoch)
                break


        if self.best_model is not None:
            self.clf = self.best_model
            print('Best model based on validation accuracy (%f) selected in epoch %d.' % (best_acc, best_epoch))

    def _train_itr(self, epoch, loader_tr, optimizer, name, scheduler=None, label=True):
        self.clf.train()
        dt_size = len(loader_tr)
        accFinal, tot_loss, iters = 0., 0., 0
        for batch_idx, (imgs, txts, labels, idxs) in enumerate(loader_tr):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels.float().cuda()
            optimizer.zero_grad()

            common_feature_image, common_feature_text, \
            image_classifier_logits, text_classifier_logits, \
            label_feature, A, \
            real_image_out, fake_image_out, \
            fake_text_out, real_text_out = self.clf(imgs, txts)

            bs = labels.shape[0]
            gan_loss = self._calc_gan_loss(real_image_out, fake_image_out, fake_text_out, real_text_out, bs)
            la = nn.L1Loss()(A, torch.eye(A.shape[0]).cuda())
            if label:
                # 判断是否使用label 计算loss
                mmd_loss = self._calc_mmd_loss(common_feature_image, common_feature_text, imgs, txts, label_feature, labels)
                _loss = self._calc_loss(common_feature_image, common_feature_text, image_classifier_logits, text_classifier_logits,
                                 labels, labels, self.args['alpha'])
                loss = _loss + 0.3 * la + self.args['beta'] * gan_loss + 0.2 * mmd_loss
            else:
                loss = 0.3 * la + self.args['beta'] * gan_loss


            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

            self.iter += 1
            iters += 1

            if scheduler is not None:
                scheduler.step(epoch - self.args['lr_warmup'] + iters/float(dt_size))

        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)
        return tot_loss / len(loader_tr)
        # return accFinal / len(loader_tr.dataset.X)
    # 100, 1000, 10000
    def train_itr(self, name, X_img, X_txt, Y, idxs_lb, X_img_val, X_txt_val, Y_val, train_epoch_func=None):  # X, Y 是所有的训练数据和label
        n_epoch = 2000 if self.args['n_epoch'] <= 0 else self.args['n_epoch']

        if self.args['optimizer'] == 'Adam':
            print('Adam optimizer...')
            optimizer = optim.Adam(self.clf.parameters(), lr=self.args['optimizer_args']['lr'], betas=(self.args['betas'][0], self.args['betas'][1]))
        else:
            print('SGD optimizer...')
            optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        optimizer.zero_grad()

        if self.args['lr_schedule']:
            for param in optimizer.param_groups:
                param['initial_lr'] = param['lr']

            scheduler = CosineAnnealingLR(optimizer, n_epoch, eta_min=0)
        else:
            scheduler = None

        idxs_train = np.arange(len(Y))[idxs_lb]
        idxs_unlb_train = np.arange(len(Y))[~idxs_lb]
        loader_tr_lb = DataLoader(self.handler(X_img[idxs_train], X_txt[idxs_train], Y[idxs_train]), shuffle=True, **self.args['loader_tr_args'])
        loader_te = DataLoader(self.handler(X_img_val, X_txt_val, Y_val), shuffle=False, **self.args['loader_te_args'])
        self.iter = 0
        self.best_model = None
        best_acc, best_epoch, n_stop = 0., 0, 0
        best_mAP, best_i2t, best_t2i = 0, 0, 0
        lr = self.args['optimizer_args']['lr']
        print('Training started...')
        lowest_loss = 1e20
        times = 0
        for epoch in tqdm(range(n_epoch)):
            # if self.args['lr_decay_epochs'] is not None and epoch in self.args['lr_decay_epochs']:
            #     for param in optimizer.param_groups:
            #         lr /= 10.
            #         param['lr'] = lr
            #         param['initial_lr'] = lr
            #
            #         if self.args['lr_schedule']:
            #             scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.args['lr_T_0'],
            #                                                     T_mult=self.args['lr_T_mult'])

            # if epoch < self.args['lr_warmup']:
            #     learning_rate = lr * (epoch + 1) / float(self.args['lr_warmup'])
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = learning_rate

            for param in optimizer.param_groups:
                self.writer.add_scalar('learning_rate/%s' % name, param['lr'], epoch)

            loss_current_label = self._train_itr(epoch, loader_tr_lb, optimizer, name,
                                                scheduler=(None if epoch < self.args['lr_warmup'] else scheduler), label=True)

            self.writer.add_scalar('training_label_loss/%s' % name, loss_current_label, epoch)

            img2txt, txt2img = self.predict(X_img=None, X_txt=None, Y=None, loader_te=loader_te, avg=False)
            avg_mAP = (img2txt + txt2img) / 2.

            self.writer.add_scalar('average_map', avg_mAP)

            print('Average MAP = {}'.format(avg_mAP))
            if avg_mAP > best_mAP:
                best_mAP, best_i2t, best_t2i = avg_mAP, img2txt, txt2img
                times = 0
            else:
                times += 1

            if not self.args['train_to_end'] and times == self.args['early_stop_times']:
                # print('Reached max accuracy at epoch %d ' % epoch)
                print('mAP has not increased in %d epochs! Stop at epoch %d, mAP is %s' %
                      (self.args['early_stop_times'], epoch, best_mAP))
                return best_mAP, best_i2t, best_t2i


        if self.best_model is not None:
            self.clf = self.best_model
            print('Best model based on lowest loss (%f) selected in epoch %d.' % (lowest_loss, best_epoch))

    def predict(self, X_img, X_txt, Y, loader_te=None, avg=True):
        if loader_te is None:
            loader_te = DataLoader(self.handler(X_img, X_txt, Y), shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        t_imgs, t_txts, t_labels = [], [], []
        img_md_img, text_md_img, img_md_text, text_md_text = 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (imgs, txts, labels, index) in enumerate(loader_te):
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    txts = txts.cuda()
                    labels = labels.float().cuda()
                t_view1_feature, t_view2_feature, _, _, _, _, real_image_out, fake_image_out, fake_text_out, real_text_out = self.clf(
                    imgs, txts)
                t_imgs.append(t_view1_feature.cpu().numpy())
                t_txts.append(t_view2_feature.cpu().numpy())
                t_labels.append(labels.cpu().numpy())
                bs = labels.shape[0]

                img_md = torch.ones(bs, dtype=torch.long).cuda()
                txt_md = torch.zeros(bs, dtype=torch.long).cuda()
                img_md_img += torch.sum(torch.argmax(real_image_out, dim=1) == img_md).cpu()
                text_md_img += torch.sum(torch.argmax(fake_image_out, dim=1) == txt_md).cpu()
                img_md_text += torch.sum(torch.argmax(fake_text_out, dim=1) == img_md).cpu()
                text_md_text += torch.sum(torch.argmax(real_text_out, dim=1) == txt_md).cpu()

        t_imgs = np.concatenate(t_imgs)
        t_txts = np.concatenate(t_txts)
        t_labels = np.concatenate(t_labels)

        img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
        txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
        if avg:
            return (img2text + txt2img) / 2.

        return img2text, txt2img

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), self.clf.n_label])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_prob_embed(self, X, Y, eval=True):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        probs = torch.zeros([len(Y), self.clf.n_label]) # 18000, 24
        embeddings = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        if eval:
            self.clf.eval()
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] = prob.cpu()
                    embeddings[idxs] = e1.cpu()
        else:
            self.clf.train()
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
                embeddings[idxs] = e1.cpu()

        return probs, embeddings


    def predict_prob_embed(self, X_img, X_txt, Y, eval=True, image=True, text=False):
        loader_te = DataLoader(self.handler(X_img, X_txt, Y), shuffle=False, **self.args['loader_te_args'])

        probs = torch.zeros([len(Y), self.clf.n_label])
        embeddings = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        if eval:
            self.clf.eval()
            with torch.no_grad():
                for batch_idxs, (x_img, x_txt, y, idxs) in enumerate(loader_te):
                    x_img, x_txt, y = x_img.to(self.device), x_txt.to(self.device), y.to(self.device)
                    common_feature_image, common_feature_text, \
                    image_classifier_logits, text_classifier_logits, \
                    label_feature, A, \
                    real_image_out, fake_image_out, \
                    fake_text_out, real_text_out = self.clf(x_img, x_txt)
                    if image and ~text:
                        probs[idxs] = image_classifier_logits.cpu()
                        embeddings[idxs] = common_feature_image.cpu()
                    elif ~image and text:
                        probs[idxs] = text_classifier_logits.cpu()
                        embeddings[idxs] = common_feature_text.cpu()
        else:
            self.clf.train()
            for batch_idxs, (x_img, x_txt, y, idxs) in enumerate(loader_te):
                x_img, x_txt, y = x_img.to(self.device), x_txt.to(self.device), y.to(self.device)
                common_feature_image, common_feature_text, \
                image_classifier_logits, text_classifier_logits, \
                label_feature, A, \
                real_image_out, fake_image_out, \
                fake_text_out, real_text_out = self.clf(x_img, x_txt)

                if image and ~text:
                    probs[idxs] = image_classifier_logits.cpu()
                    embeddings[idxs] = common_feature_image.cpu()
                elif ~image and text:
                    probs[idxs] = text_classifier_logits.cpu()
                    embeddings[idxs] = common_feature_text.cpu()

        return probs, embeddings

    def predict_all_representations(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        probs = torch.zeros([len(Y), self.clf.n_label])
        all_reps = [None for i in range(self.clf.get_n_representations())]
        embeddings = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        if eval:
            self.clf.eval()
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1, reps = self.clf(x, return_reps=True)
                    for i in range(self.clf.get_n_representations()):
                        if all_reps[i] is None:
                            all_reps[i] = reps[i]
                        else:
                            all_reps[i] = np.concatenate([all_reps[i], reps[i]], axis=0)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] = prob.cpu()
                    embeddings[idxs] = e1.cpu()

        return probs, embeddings, all_reps

    def predict_embedding_prob(self, X_embedding):
        loader_te = DataLoader(SimpleDataset(X_embedding),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([X_embedding.size(0), self.clf.n_label])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)
                out, e1 = self.clf(x, embedding=True)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), self.clf.n_label])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X_img, X_txt, Y, n_drop):
        loader_te = DataLoader(self.handler(X_img, X_txt, Y), shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), self.clf.n_label])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for batch_idxs, (x_img, x_txt, y, idxs) in enumerate(loader_te):
                    x_img, x_txt, y = x_img.to(self.device), x_txt.to(self.device), y.to(self.device)
                    common_feature_image, common_feature_text, \
                    image_classifier_logits, text_classifier_logits, \
                    label_feature, A, \
                    real_image_out, fake_image_out, \
                    fake_text_out, real_text_out = self.clf(x_img, x_txt)
                    probs[i][idxs] += F.softmax(image_classifier_logits, dim=1).cpu()
        return probs

    def predict_prob_embed_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), self.clf.n_label])
        embeddings = torch.zeros([n_drop, len(Y), self.clf.get_embedding_dim()])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
                    embeddings[i][idxs] = e1.cpu()

        return probs, embeddings

    def get_embedding(self, X_img, X_txt, Y, all_modal=False):
        loader_te = DataLoader(self.handler(X_img, X_txt, Y),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        img_embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        txt_emebdding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for batch_idxs, (x_img, x_txt, y, idxs) in enumerate(loader_te):
                x_img, x_txt, y = x_img.to(self.device), x_txt.to(self.device), y.to(self.device)
                common_feature_image, common_feature_text, \
                image_classifier_logits, text_classifier_logits, \
                label_feature, A, \
                real_image_out, fake_image_out, \
                fake_text_out, real_text_out = self.clf(x_img, x_txt)
                img_embedding[idxs] = common_feature_image.cpu()
                txt_emebdding[idxs] = common_feature_text.cpu()
        if all_modal:
            return img_embedding, txt_emebdding
        else:
            return img_embedding

    def get_grad_embedding(self, X_img, X_txt, Y, is_embedding=False):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        if is_embedding:
            loader_te = DataLoader(SimpleDataset2(X_img, X_txt, Y),
                                   shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X_img, X_txt, Y), shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            print('Creating gradient embeddings:')
            for batch_idxs, (x_img, x_txt, y, idxs) in enumerate(loader_te):
                x_img, x_txt, y = x_img.to(self.device), x_txt.to(self.device), y.to(self.device)
                common_feature_image, common_feature_text, \
                image_classifier_logits, text_classifier_logits, \
                label_feature, A, \
                real_image_out, fake_image_out, \
                fake_text_out, real_text_out = self.clf(x_img, x_txt)

                common_feature_image = common_feature_image.data.cpu().numpy()
                batchProbs = F.softmax(image_classifier_logits, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = copy.deepcopy(common_feature_image[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = copy.deepcopy(common_feature_image[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)


    def predict_similarity(self, X_img, X_txt, Y):
        loader_te = DataLoader(self.handler(X_img, X_txt, Y), shuffle=False, **self.args['loader_te_args'])
        t_imgs, t_txts, t_labels = [], [], []
        img_md_img, text_md_img, img_md_text, text_md_text = 0, 0, 0, 0
        with torch.no_grad():
            for batch_index, (imgs, txts, labels, idxs) in enumerate(loader_te):
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    txts = txts.cuda()
                    labels = labels.float().cuda()
                t_view1_feature, t_view2_feature, _, _, _, _, real_image_out, fake_image_out, fake_text_out, real_text_out = self.clf(
                    imgs, txts)
                t_imgs.append(t_view1_feature.cpu().numpy())
                t_txts.append(t_view2_feature.cpu().numpy())
                t_labels.append(labels.cpu().numpy())
                bs = labels.shape[0]

                img_md = torch.ones(bs, dtype=torch.long).cuda()
                txt_md = torch.zeros(bs, dtype=torch.long).cuda()
                img_md_img += torch.sum(torch.argmax(real_image_out, dim=1) == img_md).cpu()
                text_md_img += torch.sum(torch.argmax(fake_image_out, dim=1) == txt_md).cpu()
                img_md_text += torch.sum(torch.argmax(fake_text_out, dim=1) == img_md).cpu()
                text_md_text += torch.sum(torch.argmax(real_text_out, dim=1) == txt_md).cpu()

        t_imgs = np.concatenate(t_imgs)
        t_txts = np.concatenate(t_txts)
        t_labels = np.concatenate(t_labels)
        sim = np.matmul(t_imgs, np.transpose(t_txts))
        return sim




    def _calc_gan_loss(self, view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2, bs):
        criterion_md = nn.CrossEntropyLoss()
        img_md = torch.ones(bs, dtype=torch.long).cuda()
        txt_md = torch.zeros(bs, dtype=torch.long).cuda()
        return criterion_md(view1_modal_view1, img_md) + criterion_md(view2_modal_view1, txt_md) + \
               criterion_md(view1_modal_view2, img_md) + criterion_md(view2_modal_view2, txt_md)

    def _calc_mmd_loss(self, common_image_feature, common_text_feature, original_image_feature, original_text_feature,
                      label_feature, labels):
        image_feature = torch.cat([common_image_feature, original_image_feature], dim=1)
        text_feature = torch.cat([common_text_feature, original_text_feature], dim=1)

        image_feature = F.normalize(image_feature, dim=1)
        text_feature = F.normalize(text_feature, dim=1)
        bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

        label_feature = labels.matmul(label_feature)
        image_label_feature = torch.cat([label_feature, original_image_feature], dim=1)
        text_label_feature = torch.cat([label_feature, original_text_feature], dim=1)

        image_label_feature = F.normalize(image_label_feature, dim=1)
        text_label_feature = F.normalize(text_label_feature, dim=1)

        term1 = torch.sqrt(mix_rbf_mmd2(image_feature, image_label_feature, sigmas=bandwidths))
        term2 = torch.sqrt(mix_rbf_mmd2(text_feature, text_label_feature, sigmas=bandwidths))

        mmd_loss = term1 + term2
        return mmd_loss

    def _calc_loss(self, common_feature_image, common_feature_text, image_classifier_logits, text_classifier_logits, labels_1, labels_2, alpha):
        term1 = ((image_classifier_logits - labels_1.float()) ** 2).sum(1).sqrt().mean() + 0.5 * (
                    (text_classifier_logits - labels_2.float()) ** 2).sum(1).sqrt().mean()
        # term1 = criterion(view1_predict, labels_1) + criterion(view2_predict, labels_2)

        cos = lambda x, y: x.mm(y.t()) / (
            (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
        theta11 = cos(common_feature_image, common_feature_image)
        theta12 = cos(common_feature_image, common_feature_text)
        theta22 = cos(common_feature_text, common_feature_text)

        Sim11 = self._calc_label_sim(labels_1, labels_1).float()
        Sim12 = self._calc_label_sim(labels_1, labels_2).float()
        Sim22 = self._calc_label_sim(labels_2, labels_2).float()

        term21 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
        term22 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
        term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
        term2 = term21 + term22 + term23

        im_loss = term1 + alpha * term2
        return im_loss

    def _calc_label_sim(self, label_1, label_2):
        Sim = label_1.float().mm(label_2.float().t())
        return Sim

class SimpleDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        x = self.X[index]
        return x, index

    def __len__(self):
        return len(self.X)

class SimpleDataset2(Dataset):
    def __init__(self, X_img, X_txt, Y):
        self.X_img = X_img
        self.X_txt = X_txt
        self.Y = Y

    def __getitem__(self, index):
        x_img = self.X_img[index]
        x_txt = self.X_txt[index]
        y = self.Y[index]
        return x_img, x_txt, y, index

    def __len__(self):
        return len(self.X_img)
