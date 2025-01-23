import numpy as np
from .strategy import Strategy
import torch.nn.functional as F


class CDALSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
				 X_img = None, X_txt = None, X_img_val = None, X_txt_val = None):
		super(CDALSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
										   X_img, X_txt, X_img_val, X_txt_val)

	def query(self, n):
		self.query_count += 1
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		probs, embeddings = self.predict_prob_embed(self.X_img[idxs_unlabeled], self.X_txt[idxs_unlabeled], self.Y[idxs_unlabeled])
		probs_l, _ = self.predict_prob_embed(self.X_img[self.idxs_lb], self.X_txt[self.idxs_lb], self.Y[self.idxs_lb])

		chosen = self.model.select_coreset(F.softmax(probs, dim=1).numpy(), F.softmax(probs_l, dim=1).numpy(), n)

		return idxs_unlabeled[chosen], embeddings, None, None, None, None
