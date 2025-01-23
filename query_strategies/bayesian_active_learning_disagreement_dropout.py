import numpy as np
import torch
from .strategy import Strategy


class BALDDropout(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
				 X_img = None, X_txt = None, X_img_val = None, X_txt_val = None):
		super(BALDDropout, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
										  X_img, X_txt, X_img_val, X_txt_val)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout_split(self.X_img[idxs_unlabeled], self.X_txt[idxs_unlabeled], self.Y[idxs_unlabeled], self.args.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1

		probs, embeddings = self.predict_prob_embed(self.X_img[idxs_unlabeled],
													self.X_txt[idxs_unlabeled], self.Y[idxs_unlabeled])
		selected = U.sort()[1][:n]
		return idxs_unlabeled[selected], embeddings, probs, probs, selected, None
