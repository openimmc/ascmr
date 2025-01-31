import numpy as np
import scipy
import scipy.spatial


def fx_calc_map_label(image, text, label, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    sim = (np.dot(label, label.T) > 0).astype(float)
    tindex = np.arange(numcases, dtype=float) + 1
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        sim[i] = sim[i][order]
        num = sim[i].sum()
        a = np.where(sim[i]==1)[0]
        sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
        res += [(sim[i] / tindex).sum() / num]

    return np.mean(res)


def fx_calc_map_label_per_class(image, text, label, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    sim = (np.dot(label, label.T) > 0).astype(float)
    tindex = np.arange(numcases, dtype=float) + 1
    res = []
    for i in range(numcases):
        order = ord[i]
        sim[i] = sim[i][order]
        num = sim[i].sum()
        a = np.where(sim[i]==1)[0]
        sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
        res += [(sim[i] / tindex).sum() / num]

    categories = label.shape[1]
    per_class = np.zeros([categories], dtype=np.float)
    for i in range(numcases):
        per_class += res[i] * label[i]

    per_class /= label.sum(0)

    return per_class


def fuse_matrix(image, text, label, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    lab_sum = label.sum(0)
    numcases = dist.shape[0]
    classes = label.shape[1]
    fuse = np.zeros((classes, classes))
    for i in range(numcases):
        for j in range(classes):
            if label[i, j] == 1:
                for k in range(lab_sum[j]):
                    for m in range(classes):
                        if label[ord[i, k] ,m] == 1:
                            fuse[j, m] += 1

    fuse /= lab_sum[:, None] ** 2

    return fuse
