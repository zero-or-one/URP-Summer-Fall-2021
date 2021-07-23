import numpy as np
import torch
from torch.nn import functional as F

def KMeans():

    def __init__(self, cluster_num=2):
        self.cluster_num = cluster_num
        self.centroids = None

    def init_centroids(self, data):
        size = data.shape[0]
        dim = data.shape[1]
        # range of centroids
        max_range = data.max(dim=0)[0].unsqueeze(1)
        min_range = data.min(dim=0)[0].unsqueeze(1)
        centroids = (min_range - max_range) * torch.rand((dim, self.n_clusters)) + max_range
        return centroids

    def assign_cluster(self, data, centroids):
        size = data.shape[0]
        d = torch.zeros((size, self.cluster_num))
        for i in range(self.cluster_num):
            cent = centroids[:, i].unsqueeze(1)
            d[:, i] = (data - cent.T).pow(2).sum(dim=1).sqrt()
        return d

    def update_centroids(self, data, centroids, lab):
        for i in range(self.n_clusters):  # M-step: re-compute the centroids
            ind = torch.where(lab == i)[0]
            if len(ind) == 0:
                continue
            centroids[:, i] = data[ind].mean(dim=0)
        return centroids

    def fit(self, data, epoch=21):
        minLoss = 1e5
        pred = None
        for _ in range(epoch):
            centroids = self.init_centroids(data)
            cLoss = -1
            pLoss = -1
            while True:
                d_min, fam = self.assign_cluster(data, centroids).min(dim=1)
                centroids = self.update_centroids(data, centroids, fam)
                cLoss = d_min.sum()
                if pLoss == cLoss: # no change happens
                    break
                pLoss = cLoss
            if cLoss < minLoss:
                minLoss = cLoss
                pred = fam
        return pred

