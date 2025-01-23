from copy import deepcopy


class Strategy:
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
                 X_img=None, X_txt=None, X_img_val=None, X_txt_val=None):
        self.X = X  # train input
        self.Y = Y  # train label
        self.X_val = X_val  # []
        self.Y_val = Y_val  # []
        self.idxs_lb = idxs_lb
        self.device = device
        self.model = model  # Training
        self.args = args
        self.n_pool = len(Y)    # 60000

        self.writer = writer

        self.query_count = 0
        self.X_img, self.X_txt, self.X_img_val, self.X_txt_val = X_img, X_txt, X_img_val, X_txt_val


    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def train(self, name):
        if self.args.task == 'ITR':
            # self, name, X_img, X_txt, Y, idxs_lb, X_img_val, X_txt_val, train_epoch_func=None
            return self.model.train_itr(name, self.X_img, self.X_txt, self.Y, self.idxs_lb, self.X_img_val, self.X_txt_val, self.Y_val)
        else:
            return self.model.train(name, self.X, self.Y, self.idxs_lb, self.X_val, self.Y_val)

    def predict(self, X_img, X_txt, Y, loader_te=None, avg=True):
        return self.model.predict(X_img, X_txt, Y, loader_te=loader_te, avg=avg)

    def predict_prob(self, X, Y):
        return self.model.predict_prob(X, Y)

    # def predict_prob_embed(self, X, Y, eval=True):
    #     return self.model.predict_prob_embed(X, Y, eval)
    # entropy sampling
    def predict_prob_embed(self, X_img, X_txt, Y, eval=True, image=True, text=False):
        return self.model.predict_prob_embed(X_img, X_txt, Y, eval, image, text)

    def predict_all_representations(self, X, Y):
        return self.model.predict_all_representations(X, Y)

    def predict_embedding_prob(self, X_embedding):
        return self.model.predict_embedding_prob(X_embedding)

    def predict_prob_dropout(self, X, Y, n_drop):
        return self.model.predict_prob_dropout(X, Y, n_drop)

    # def predict_prob_dropout_split(self, X, Y, n_drop):
    #     return self.model.predict_prob_dropout_split(X, Y, n_drop)
    # BALD
    def predict_prob_dropout_split(self, X_img, X_txt, Y, n_drop):
        return self.model.predict_prob_dropout_split(X_img, X_txt, Y, n_drop)

    def predict_prob_embed_dropout_split(self, X, Y, n_drop):
        return self.model.predict_prob_embed_dropout_split(X, Y, n_drop)

    # def get_embedding(self, X, Y):
    #     return self.model.get_embedding(X, Y)
    # CoreSet
    def get_embedding(self, X_img, X_txt, Y, all_modal=False):
        return self.model.get_embedding(X_img, X_txt, Y, all_modal)

    # def get_grad_embedding(self, X, Y, is_embedding=False):
    #     return self.model.get_grad_embedding(X, Y, is_embedding)
    # BADGE
    def get_grad_embedding(self, X_img, X_txt, Y, is_embedding=False):
        return self.model.get_grad_embedding(X_img, X_txt, Y, is_embedding)

    # Contrastive
    def predict_similarity(self, X_img, X_txt, Y):
        return self.model.predict_similarity(X_img, X_txt, Y)