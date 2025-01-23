import numpy as np
import torch
from .strategy import Strategy
class InterModalitySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
				 X_img = None, X_txt = None, X_img_val = None, X_txt_val = None):
		super(InterModalitySampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
											 X_img, X_txt, X_img_val, X_txt_val)
		self.s_margin = 0.1

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		unlb_img_embeds, unlb_txt_embeds = self.X_img[idxs_unlabeled], self.X_txt[idxs_unlabeled]
		lb_img_embeds, lb_txt_embeds = self.X_img[self.idxs_lb], self.X_txt[self.idxs_lb]
		Y_lb, Y_unlb = self.Y[self.idxs_lb], self.Y[idxs_unlabeled]
		features_img = np.concatenate([unlb_img_embeds, lb_img_embeds], axis=0)
		features_txt = np.concatenate([unlb_txt_embeds, lb_txt_embeds], axis=0)
		Y = np.concatenate([Y_unlb, Y_lb], axis=0)
		nlbl = np.arange(0, unlb_img_embeds.shape[0], 1)

		sims = self.predict_similarity(features_img, features_txt, Y)
		selected = np.argsort(np.diag(sims))[:n]
		return idxs_unlabeled[selected], None, None, None, selected, None
