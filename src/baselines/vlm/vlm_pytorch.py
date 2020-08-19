import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
from torch import nn

# Loosely adapted from Ehtsham Elahi's original Tensorflow implementation
# https://github.com/ehtsham/recsys19vlm/blob/master/RecSys2019-VLMPaper.ipynb

class VLM_PyTorch(nn.Module):
    def __init__(self, args):
        super(VLM_PyTorch, self).__init__()

        self.num_users = args['num_users']
        self.num_items = args['num_items']
        self.num_tags = args['num_tags']
        self.num_factors = args['num_factors']
        self.var_prior = args['var_prior']
        self.reg = args['reg']
        self.side_info = args['side_info']
        if self.side_info:
            # Experimental- Sparse matrix to speed up multiplication
            print('Setting up embeddings for tags...')
            item_tag_mat = args['item_tag_mat'].tocoo()
            values = item_tag_mat.data
            indices = np.vstack((item_tag_mat.row, item_tag_mat.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = item_tag_mat.shape
            self.item_tag_mat = torch.nn.Parameter(torch.sparse.FloatTensor(i, v, torch.Size(shape)), requires_grad = False)
            # Deprecated - Dense matrix makes multiplication too slow to handle
            # self.item_tag_mat = torch.nn.Parameter(torch.from_numpy(args['item_tag_mat']), requires_grad = False) #.to(args['device'])
            self.Mu_Zt = nn.Embedding(self.num_tags,  self.num_factors) # Mean latent factors for tags

        self.Mu_Zu = nn.Embedding(self.num_users, self.num_factors) # Mean latent factors for users
        self.lsdev_Zu = nn.Embedding(self.num_users, 1)             # Log(std-deviation) for user latent factors
        self.Mu_Zv = nn.Embedding(self.num_items, self.num_factors) # Mean latent factors for items

    def forward(self, user_ids, add_noise = True):
        # Get mean and log(std-dev) for users in this batch
        Mu_Zu_batch, lsdev_Zu_batch = self.Mu_Zu(user_ids), self.lsdev_Zu(user_ids)

        # Map item-tag matrix to tag embeddings
        if self.side_info:
            Mu_Zv_hat = torch.mm(self.item_tag_mat, self.Mu_Zt.weight)
            # TODO - validate whether sparse matrix multiplication is faster than gathering and summing embeddings

        # Simple things first - let's not bring in side-info just yet
        if add_noise:
            # 'Reparameterisation trick' - sample Gaussian noise over factors
            eps = torch.randn_like(Mu_Zu_batch)
            
            # Bring it together - mean + eps * std-dev
            Zu_batch = Mu_Zu_batch + eps * torch.exp(lsdev_Zu_batch)

            # Compute scores between both as the dot product
            if self.side_info:
                batch_logits = torch.mm(Zu_batch, self.Mu_Zv.weight.T + Mu_Zv_hat.T)
            else:
                batch_logits = torch.mm(Zu_batch, self.Mu_Zv.weight.T)
        else:
            if self.side_info:
                batch_logits = torch.mm(Mu_Zu_batch, self.Mu_Zv.weight.T + Mu_Zv_hat.T)
            else:
                batch_logits = torch.mm(Mu_Zu_batch, self.Mu_Zv.weight.T)

        log_softmax = torch.nn.functional.log_softmax(batch_logits, dim = 1)
        return log_softmax, Mu_Zu_batch, lsdev_Zu_batch
