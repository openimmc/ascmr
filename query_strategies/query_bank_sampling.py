import numpy as np
import torch

from .strategy import Strategy


class QueryBankSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
				 X_img = None, X_txt = None, X_img_val = None, X_txt_val = None):
		super(QueryBankSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
											 X_img, X_txt, X_img_val, X_txt_val)


	def query(self, n):
		self.query_count += 1
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		img_embedding_lb = self.get_embedding(self.X_img[self.idxs_lb], self.X_txt[self.idxs_lb], self.Y[self.idxs_lb])
		img_embedding_ulb = self.get_embedding(self.X_img[idxs_unlabeled], self.X_txt[idxs_unlabeled], self.Y[idxs_unlabeled])
		txt_embedding_lb = self.get_txt_embedding(self.X_img[self.idxs_lb], self.X_txt[self.idxs_lb], self.Y[self.idxs_lb])
		txt_embedding_ulb = self.get_txt_embedding(self.X_img[idxs_unlabeled], self.X_txt[idxs_unlabeled], self.Y[idxs_unlabeled])


		sim_lb_txt_ulb_img = torch.matmul(txt_embedding_lb, img_embedding_ulb.t())
		sim_ulb_txt_ulb_img = torch.matmul(txt_embedding_ulb, img_embedding_ulb.t())
		idx_lb = self.qb_norm(sim_lb_txt_ulb_img.cpu().numpy(), sim_ulb_txt_ulb_img.cpu().numpy(), n)

		selected = torch.Tensor(idx_lb).diag().sort()[1][:n].detach().cpu().numpy()
		return selected, None, None, None, None, None

	# Returns list of retrieved top k videos based on the sims matrix
	def get_retrieved_videos(self, sims, k):
		argm = np.argsort(-sims, axis=1)
		topk = argm[:, :k].reshape(-1)
		retrieved_videos = np.unique(topk)
		return retrieved_videos

	# Returns list of indices to normalize from sims based on videos
	def get_index_to_normalize(self, sims, videos):
		argm = np.argsort(-sims, axis=1)[:, 0]
		result = np.array(list(map(lambda x: x in videos, argm)))
		result = np.nonzero(result)
		return result

	def qb_norm(self, train_test, test_test, n):
		k = 1
		beta = 1
		retrieved_videos = self.get_retrieved_videos(train_test, k)  # 618
		test_test_normalized = test_test

		normalizing_sum = np.sum(train_test, axis=0)
		index_for_normalizing = self.get_index_to_normalize(test_test, retrieved_videos)
		test_test_normalized[index_for_normalizing, :] = \
			np.divide(test_test[index_for_normalizing, :], normalizing_sum)
		return test_test_normalized