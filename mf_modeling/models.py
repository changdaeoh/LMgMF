'''
    borrowed from https://github.com/khanhnamle1994/MetaRec
'''

import torch
from torch import nn, distributions
import torch.nn.functional as F
import pdb
import numpy as np

def l2_regularize(array):
    """
    Function to do L2 regularization
    """
    loss = torch.sum(array ** 2.0)
    return loss

user_gindex_dict = {
    1: 1216,
    2: 2513,
    3: 3862,
    4: 5317,
    5: 6474,
    6: 7624,
    7: 8554,
    8: 9404,
    9: 9833
}

item_gindex_dict = {
    1: 974,
    2: 1996,
    3: 3056,
    4: 4087,
    5: 5353,
    6: 6566,
    7: 7888,
    8: 8938,
    9: 9453
}

def grade_slicing(src_emb, idx_dict, grade, dataset='benchmark'):
    if dataset == 'benchmark':
        return src_emb

    if grade == 'entire':
        return src_emb
    elif grade == '1':
        tar_emb = src_emb[:idx_dict[int(grade)]+1]
    else:
        tar_emb = src_emb[idx_dict[int(grade)-1]+1:idx_dict[int(grade)]+1]
    return tar_emb

# Vanilla MF
class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_user, n_item, writer, device, initpath, method, dataset, **kwargs):
        """
        :param n_user: User column
        :param n_item: Item column
        :param ld: Dimensions constant
        :param reg_feat: Regularization constant
        :param reg_bias: Regularization constant for the biases
        :param writer: Log results via TensorBoard
        """
        super(MF, self).__init__()
        self.device = device

        # This will hold the logging
        self.writer = writer
        
        # These are the hyper-parameters
        self.ld = kwargs['ld']
        self.n_user = n_user
        self.n_item = n_item
        self.reg_bias = kwargs['reg_bias']
        self.reg_feat = kwargs['reg_feat']
        self.method = method

        # if n_item > 10000:
        #     pdb.set_trace()
            
        # The embedding matrices for user and item are learned and fit by PyTorch
        self.user = nn.Embedding(n_user, kwargs['ld'])
        self.item = nn.Embedding(n_item, kwargs['ld'])
        #pdb.set_trace()
        if len(initpath) == 2:
            if method == 'lminit_u':
                u_teacher = torch.tensor(np.load(initpath[0]), dtype=torch.float32)/4.0
                self.user.weight.data = grade_slicing(u_teacher, user_gindex_dict, kwargs['grade'], dataset)
            elif method == 'lminit_i':
                i_teacher = torch.tensor(np.load(initpath[1]), dtype=torch.float32)/4.0
                self.item.weight.data = grade_slicing(i_teacher, item_gindex_dict, kwargs['grade'], dataset)
            elif method == 'lminit_ui':
                u_teacher = torch.tensor(np.load(initpath[0]), dtype=torch.float32)/4.0
                i_teacher = torch.tensor(np.load(initpath[1]), dtype=torch.float32)/4.0
                self.user.weight.data = grade_slicing(u_teacher, user_gindex_dict, kwargs['grade'], dataset)
                self.item.weight.data = grade_slicing(i_teacher, item_gindex_dict, kwargs['grade'], dataset)
            elif method == 'lmdistill_u':
                u_teacher = torch.tensor(np.load(initpath[0]), dtype=torch.float32, requires_grad=False)/4.0
                self.u_teacher = grade_slicing(u_teacher, user_gindex_dict, kwargs['grade'], dataset).to(self.device)
                del u_teacher
                #self.u_proj = nn.Linear(self.u_teacher.shape[1], self.ld)
                self.u_proj = nn.Linear(self.u_teacher.shape[1], 1)
                self.dist_u = kwargs['dist_u']
            elif method == 'lmdistill_i':
                i_teacher = torch.tensor(np.load(initpath[1]), dtype=torch.float32, requires_grad=False)/4.0
                self.i_teacher = grade_slicing(i_teacher, item_gindex_dict, kwargs['grade'], dataset).to(self.device)
                del i_teacher
                #self.i_proj = nn.Linear(self.i_teacher.shape[1], self.ld)
                self.i_proj = nn.Linear(self.i_teacher.shape[1], 1)
                self.dist_i = kwargs['dist_i']
            elif method == 'lmdistill_ui':
                u_teacher = torch.tensor(np.load(initpath[0]), dtype=torch.float32, requires_grad=False)/4.0
                self.u_teacher = grade_slicing(u_teacher, user_gindex_dict, kwargs['grade'], dataset).to(self.device)
                del u_teacher
                #self.u_proj = nn.Linear(self.u_teacher.shape[1], self.ld)
                self.u_proj = nn.Linear(self.u_teacher.shape[1], 1)
                self.dist_u = kwargs['dist_u']
                i_teacher = torch.tensor(np.load(initpath[1]), dtype=torch.float32, requires_grad=False)/4.0
                self.i_teacher = grade_slicing(i_teacher, item_gindex_dict, kwargs['grade'], dataset).to(self.device)
                del i_teacher
                #self.i_proj = nn.Linear(self.i_teacher.shape[1], self.ld)
                self.i_proj = nn.Linear(self.i_teacher.shape[1], 1)
                self.dist_i = kwargs['dist_i']
            else:
                raise ValueError()
        else:
            # nn.init.normal_(self.user.weight.data, std=1.0)
            # nn.init.normal_(self.item.weight.data, std=1.0)
            pass

        # We've added new terms here: Embedding matrices for the user's biases and the item's biases
        self.bias_user = nn.Embedding(n_user, 1) #.to(device)
        self.bias_item = nn.Embedding(n_item, 1) #.to(device)
        # nn.init.normal_(self.bias_user.weight.data, std=1.0)
        # nn.init.normal_(self.bias_item.weight.data, std=1.0)

        # Initialize the bias tensors
        self.bias = nn.Parameter(torch.ones(1)*1.0)

    def __call__(self, train_x):
        """This is the most important function in this script"""
        #pdb.set_trace()
        # These are the user indices, and correspond to "u" variable
        user_id = train_x[:, 0].to(self.device)
        self.batch_user_id = user_id
        # These are the item indices, correspond to the "i" variable
        item_id = train_x[:, 1].to(self.device)
        self.batch_item_id = item_id

        # Initialize a vector user = p_u using the user indices
        vector_user = self.user(user_id)
        # Initialize a vector item = q_i using the item indices
        vector_item = self.item(item_id)

        # The user-item interaction: p_u * q_i is a dot product between the 2 vectors above
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)

        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        # Add the bias to the user-item interaction to obtain the final prediction
        prediction = ui_interaction + biases
        return prediction

    def loss(self, prediction, target):
        """
        Function to calculate the loss metric
        """
        # Calculate the BCE loss between target and prediction
        #pdb.set_trace()
        loss_bce = F.binary_cross_entropy_with_logits(prediction, target.squeeze().float().to(self.device))

        # Compute L2 regularization over the biases for user and the biases for item matrices
        prior_bias_user = l2_regularize(self.bias_user.weight[self.batch_user_id]) * self.reg_bias
        prior_bias_item = l2_regularize(self.bias_item.weight[self.batch_item_id]) * self.reg_bias

        # Compute L2 regularization over user (P) and item (Q) matrices
        prior_user = l2_regularize(self.user.weight[self.batch_user_id]) * self.reg_feat
        prior_item = l2_regularize(self.item.weight[self.batch_item_id]) * self.reg_feat

        # Distillation
        distill = 0.0
        
        #! broadcasting
        if self.method == 'lmdistill_u':
            distill = self.dist_u * l2_regularize(self.bias_user.weight[self.batch_user_id] - self.u_proj(self.u_teacher[self.batch_user_id]))
        elif self.method == 'lmdistill_i':
            distill = self.dist_i * l2_regularize(self.bias_item.weight[self.batch_item_id] - self.i_proj(self.i_teacher[self.batch_item_id]))
        elif self.method == 'lmdistill_ui':
            distill = self.dist_u * l2_regularize(self.bias_user.weight[self.batch_user_id] - self.u_proj(self.u_teacher[self.batch_user_id]))
            distill += self.dist_i * l2_regularize(self.bias_item.weight[self.batch_item_id] - self.i_proj(self.i_teacher[self.batch_item_id]))

        # Add up the MSE loss + user & item regularization + user & item biases regularization
        total = loss_bce + prior_user + prior_item + prior_bias_user + prior_bias_item + distill

        # This logs all local variables to tensorboard
        for name, var in locals().items():
            if type(var) is torch.Tensor and var.nelement() == 1 and self.writer is not None:
                self.writer.add_scalar(name, var, self.itr)
        return total

# LM embedding-based classifier
class EmbClassifier(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, writer, device, initpath, method, clf_type):
        """
        clf_type: MLP or Linear
        """
        super(EmbClassifier, self).__init__()
        self.device = device
        self.writer = writer
        self.method = method
        
        assert len(initpath) == 2

        if method in ['lmlr_u','lmmlp_u']:
            self.u_emb = torch.tensor(np.load(initpath[0]), dtype=torch.float32).to(device)/4.0
            self.ld = self.u_emb.shape[1]
        elif method in ['lmlr_i','lmmlp_i']:
            self.i_emb = torch.tensor(np.load(initpath[1]), dtype=torch.float32).to(device)/4.0
            self.ld = self.i_emb.shape[1]
        elif method in ['lmlr_ui','lmmlp_ui']:
            self.u_emb = torch.tensor(np.load(initpath[0]), dtype=torch.float32).to(device)/4.0
            self.i_emb = torch.tensor(np.load(initpath[1]), dtype=torch.float32).to(device)/4.0
            self.ld = self.u_emb.shape[1] + self.i_emb.shape[1]
        else:
            raise ValueError
        
        if method in ['lmlr_u','lmlr_i','lmlr_ui']:
            self.clf = nn.Linear(self.ld, 1)
        else:
            self.clf = nn.Sequential(
                                    nn.Linear(self.ld, 400),
                                    nn.ReLU(),
                                    nn.Linear(400, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 1),
                                    )

    def __call__(self, train_x):
        user_id = train_x[:, 0].to(self.device)
        item_id = train_x[:, 1].to(self.device)

        if self.method in ['lmlr_u','lmmlp_u']:
            vector_user = self.u_emb[user_id]
            prediction = self.clf(vector_user)
        elif self.method in ['lmlr_i','lmmlp_i']:
            vector_item = self.i_emb[item_id]
            prediction = self.clf(vector_item)
        else:
            vector_user = self.u_emb[user_id]
            vector_item = self.i_emb[item_id]
            prediction = self.clf(torch.cat((vector_user, vector_item), axis=1))

        return prediction.squeeze()

    def loss(self, prediction, target):
        loss_bce = F.binary_cross_entropy_with_logits(prediction, target.squeeze().float().to(self.device))

        for name, var in locals().items():
            if type(var) is torch.Tensor and var.nelement() == 1 and self.writer is not None:
                self.writer.add_scalar(name, var, self.itr)
        return loss_bce


LINK = torch.abs
#LINK = F.softplus
N_VARIATIONAL_SAMPLES = 1


class VFM(nn.Module):
    """
    https://github.com/jilljenn/vae/blob/master/vfm-torch.py
    """
    def __init__(self, data_tr, data_te, embedding_size, device, kl_coef, output='class'):
        super().__init__()
        dataset = np.concatenate((data_tr, data_te), axis=0)
        N = len(np.unique(dataset[:,0]))
        M = len(np.unique(dataset[:,1]))
        self.N = N
        self.M = M
        self.device = device
        
        self.nb_occ = torch.bincount(torch.LongTensor(data_tr).flatten()).to(device)
        self.n_tr_samples = len(data_tr)
        self.embedding_size = embedding_size
        self.kl_coef = kl_coef

        self.output = output
        self.alpha = nn.Parameter(torch.Tensor([1e9]), requires_grad=True)
        self.global_bias_mean = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        self.global_bias_scale = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_global_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_item_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_entity_prior = nn.Parameter(torch.ones(embedding_size), requires_grad=True)
        self.prec_item_entity_prior = nn.Parameter(torch.ones(embedding_size), requires_grad=True)
        # self.alpha = torch.Tensor([1.])
        nn.init.uniform_(self.alpha)

        # bias_init = torch.cat((torch.randn(N + M, 1), torch.ones(N + M, 1) * (0.02 ** 0.5)), axis=1)
        # entity_init = torch.cat((
        #     torch.randn(N + M, embedding_size),
        #     torch.ones(N + M, embedding_size) * (0.02 ** 0.5),
        # ), axis=1)
        self.bias_params = nn.Embedding(N + M, 2)#.from_pretrained(bias_init)  # w
        self.entity_params = nn.Embedding(N + M, 2 * embedding_size)#.from_pretrained(entity_init)  # V

        self.saved_global_biases = []
        self.saved_mean_biases = []
        self.saved_mean_entities = []
        self.mean_saved_global_biases = None
        self.mean_saved_mean_biases = None
        self.mean_saved_mean_entities = None

        self.global_bias_prior = distributions.normal.Normal(0, 1)
        self.bias_prior = distributions.normal.Normal(0, 1)
        self.entity_prior = distributions.normal.Normal(0, 1)
        #     torch.zeros(N + M),
        #     torch.nn.functional.softplus(torch.cat((
        #         self.prec_user_bias_prior.repeat(N),
        #         self.prec_item_bias_prior.repeat(M)
        #     )))
        # )
        # self.entity_prior = distributions.normal.Normal(
        #     torch.zeros(N + M, EMBEDDING_SIZE),
        #     torch.nn.functional.softplus(torch.cat((
        #         self.prec_user_entity_prior.repeat(N, 1),
        #         self.prec_item_entity_prior.repeat(M, 1)
        #     )))
        # )

    def save_weights(self):
        self.saved_global_biases.append(self.global_bias_mean.detach().numpy().copy())
        self.saved_mean_biases.append(self.bias_params.weight[:, 0].detach().numpy().copy())
        self.saved_mean_entities.append(self.entity_params.weight[:, :self.embedding_size].detach().numpy().copy())
        self.mean_saved_global_biases = np.array(self.saved_global_biases).mean(axis=0)
        self.mean_saved_mean_biases = np.array(self.saved_mean_biases).mean(axis=0)
        self.mean_saved_mean_entities = np.array(self.saved_mean_entities).mean(axis=0)
        # print('size of saved', np.array(self.saved_mean_entities).shape)
        # print('test', np.array(self.saved_mean_biases)[:3, 0])

    def __call__(self, x):
        x = x.to(self.device)
        #pdb.set_trace()
        uniq_entities, entity_pos, nb_occ_in_batch = torch.unique(x, return_inverse=True, return_counts=True)
        uniq_users, nb_occ_user_in_batch = torch.unique(x[:, 0], return_counts=True)
        uniq_items, nb_occ_item_in_batch = torch.unique(x[:, 1], return_counts=True)
        
        # nb_occ_user_in_batch = nb_occ_user_in_batch.to(x.device)
        # nb_occ_item_in_batch = nb_occ_item_in_batch.to(x.device)
        # uniq_users = uniq_users.to(x.device)
        # uniq_users = uniq_users.to(x.device)

        # nb_uniq_users = len(uniq_users)
        # nb_uniq_items = len(uniq_items)
        # print('uniq', uniq_entities.shape, 'pos', entity_pos.shape)

        # self.global_bias_prior = distributions.normal.Normal(
        #     torch.Tensor([0.]), torch.nn.functional.softplus(self.prec_global_bias_prior))
        # Global bias
        global_bias_sampler = distributions.normal.Normal(
            self.global_bias_mean,
            LINK(self.global_bias_scale)
        )
        # Biases and entities
        bias_batch = self.bias_params(x)
        entity_batch = self.entity_params(x)
        uniq_bias_batch = self.bias_params(uniq_entities)#.reshape(-1, 2)
        uniq_entity_batch = self.entity_params(uniq_entities)#.reshape(-1, 2 * EMBEDDING_SIZE)
        # print('first', bias_batch.shape, entity_batch.shape)
        # print('samplers', uniq_bias_batch.shape, uniq_entity_batch.shape)
        # scale_bias = torch.ones_like(scale_bias) * 1e-6
        bias_sampler = distributions.normal.Normal(
            uniq_bias_batch[:, 0],
            LINK(uniq_bias_batch[:, 1])
        )
        # user_bias_posterior = distributions.normal.Normal(
        #     bias_batch[:, :, 0],
        #     LINK(bias_batch[:, :, 1])
        # )
        # diag_scale_entity = nn.functional.softplus(entity_batch[:, EMBEDDING_SIZE:])
        # diag_scale_entity = torch.ones_like(diag_scale_entity) * 1e-6
        # print('scale entity', entity_batch.shape, scale_entity.shape)
        entity_sampler = distributions.normal.Normal(
            loc=uniq_entity_batch[:, :self.embedding_size],
            scale=LINK(uniq_entity_batch[:,self.embedding_size:])
        )
        # entity_posterior = distributions.normal.Normal(
        #     loc=entity_batch[:, :, :EMBEDDING_SIZE],
        #     scale=LINK(entity_batch[:, :, EMBEDDING_SIZE:])
        # )
        # self.entity_prior = distributions.normal.Normal(
        #     loc=torch.zeros_like(entity_batch[:, :, :EMBEDDING_SIZE]),
        #     scale=torch.ones_like(entity_batch[:, :, :EMBEDDING_SIZE])
        # )

        # print('batch shapes', entity_sampler.batch_shape, self.entity_prior.batch_shape)
        # print('event shapes', entity_sampler.event_shape, self.entity_prior.event_shape)
        global_bias = global_bias_sampler.rsample((N_VARIATIONAL_SAMPLES,))
        biases = bias_sampler.rsample((N_VARIATIONAL_SAMPLES,))#.reshape(
            # N_VARIATIONAL_SAMPLES, -1, 2)
        entities = entity_sampler.rsample((N_VARIATIONAL_SAMPLES,))#.reshape(
            # N_VARIATIONAL_SAMPLES, -1, 2, EMBEDDING_SIZE)  # N_VAR_SAMPLES x BATCH_SIZE x 2 (user, item) x EMBEDDING_SIZE
        # print('hola', biases.shape, entities.shape)
        sum_users_items_biases = biases[:, entity_pos].sum(axis=2).mean(axis=0).squeeze()
        users_items_emb = entities[:, entity_pos].prod(axis=2).sum(axis=2).mean(axis=0)
        # print('final', sum_users_items_biases.shape, users_items_emb.shape)

        if self.mean_saved_mean_biases is not None:
            last_global_bias = self.saved_global_biases[-1]
            last_bias_term = self.saved_mean_biases[-1][x].sum(axis=1).squeeze()
            last_embed_term = self.saved_mean_entities[-1][x].prod(axis=1).sum(axis=1)

            mean_global_bias = self.mean_saved_global_biases
            mean_bias_term = self.mean_saved_mean_biases[x].sum(axis=1).squeeze()
            mean_embed_term = self.mean_saved_mean_entities[x].prod(axis=1).sum(axis=1)
            # print(self.mean_saved_mean_biases[x].shape, mean_bias_term.shape)
            # print(self.mean_saved_mean_entities[x].shape, mean_embed_term.shape)
            last_logits = last_global_bias + last_bias_term + last_embed_term
            mean_logits = mean_global_bias + mean_bias_term + mean_embed_term #  + 
        else:
            last_logits = None
            mean_logits = None

        std_dev = torch.sqrt(1 / LINK(self.alpha))
        unscaled_pred = global_bias + sum_users_items_biases + users_items_emb

        if self.output == 'reg':
            likelihood = distributions.normal.Normal(unscaled_pred, std_dev)
        else:
            likelihood = distributions.bernoulli.Bernoulli(logits=unscaled_pred)
        # print('global bias sampler', global_bias_sampler)
        # print('global bias prior', self.global_bias_prior)
        # print('bias sampler', bias_sampler)
        # print('bias prior', self.bias_prior)
        # print('entity sampler', entity_sampler)
        # print('entity prior', self.entity_prior)
        # a = distributions.normal.Normal(torch.zeros(2, 3), torch.ones(2, 3))
        # b = distributions.normal.Normal(torch.zeros(2, 3), torch.ones(2, 3))
        # print('oh hey', distributions.kl.kl_divergence(a, b))
        # print('oh hey', distributions.kl.kl_divergence(entity_sampler, entity_sampler))
        # print('oh hiya', distributions.kl.kl_divergence(entity_sampler, self.entity_prior))
        # print('oh hey', distributions.kl.kl_divergence(self.entity_prior, self.entity_prior))

        # print(
        #     distributions.kl.kl_divergence(global_bias_sampler, self.global_bias_prior).shape,
        #     distributions.kl.kl_divergence(bias_sampler, self.bias_prior).sum(axis=1).shape,
        #     distributions.kl.kl_divergence(entity_sampler, self.entity_prior).sum(axis=[1, 2]).shape#.sum(axis=3).shape,#.sum(axis=2).shape
        # )

        kl_bias = distributions.kl.kl_divergence(bias_sampler, self.bias_prior)
        # print('kl bias', kl_bias.shape)
        # print('bias sampler', bias_sampler)
        # print('entity sampler', entity_sampler)
        # print('entity prior', self.entity_prior)
        kl_entity = distributions.kl.kl_divergence(entity_sampler, self.entity_prior).sum(axis=1)
        # print('kl entity', kl_entity.shape)

        nb_occ_in_train = self.nb_occ[uniq_entities]
        nb_occ_user_in_train = self.nb_occ[uniq_users]
        nb_occ_item_in_train = self.nb_occ[uniq_items]
        # nb_occ_batch = torch.bincount(x.flatten())
        # print('nboccs', nb_occ_in_batch.shape, nb_occ_in_train.shape)
        # nb_occ_batch[x]

        user_normalizer = (nb_occ_user_in_batch / nb_occ_user_in_train).sum(axis=0)
        item_normalizer = (nb_occ_item_in_batch / nb_occ_item_in_train).sum(axis=0)
        # print('normalizers', user_normalizer.shape, item_normalizer.shape)

        # print('begin', ((kl_bias + kl_entity) * (nb_occ_in_batch / nb_occ_in_train)).shape)
        # print('ent', x)
        # print('ent', x <= N)
        # print('ent', (x <= N) * N)
        #pdb.set_trace()

        kl_rescaled = (
            (kl_bias + kl_entity) * (nb_occ_in_batch / nb_occ_in_train) *
            ((uniq_entities <= self.N) * self.N / user_normalizer + (uniq_entities > self.N) * self.M / item_normalizer)
        ).sum(axis=0)
        # print('rescaled', kl_rescaled.shape)

        return (likelihood,
            # last_logits, mean_logits,
            distributions.kl.kl_divergence(global_bias_sampler, self.global_bias_prior) +
            kl_rescaled
        )
    
    def loss(self, likelihood, kl_term, target):
        # pdb.set_trace()
        return - likelihood.log_prob(target.float()).mean() * self.n_tr_samples + self.kl_coef * kl_term



class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, device, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        self.device = device

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                    a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                            self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                            self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                            self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                            self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight, 
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                        self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def __call__(self, train_x):
        user = train_x[:, 0].to(self.device)
        item = train_x[:, 1].to(self.device)

        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def loss(self, prediction, target):
        loss_bce = F.binary_cross_entropy_with_logits(prediction, target.squeeze().float().to(self.device))
        return loss_bce

class NGCF(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(NGCF, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.ld
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.device = torch.device(f"cuda:{args_config.gpu}")
        self.K = args_config.K

        self.alpha=args_config.alpha

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] * (self.context_hops+1)
        for k in range(self.context_hops):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def __call__(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def gcn(self, edge_dropout=True, mess_dropout=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.edge_dropout_rate,
                                    self.sparse_norm_adj._nnz()) if edge_dropout else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        # all_embeddings = []
        all_embeddings = [ego_embeddings]

        for k in range(self.context_hops):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            if mess_dropout:
                ego_embeddings = nn.Dropout(self.mess_dropout_rate)(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)  # [n_entity, n_hops+1, emb_size]
        return all_embeddings[:self.n_users, :], all_embeddings[self.n_users:, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(edge_dropout=False, mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        # 使用我们的方法进行采样
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()  # torch.Size([2048, 3])
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        neg_items_embedding_hardest = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices, :]   #   [batch_size, n_hops+1, channel]

        neg_scores = torch.exp(s_e *neg_items_embedding_hardest)  # [batch_size, n_hops, channel]
        total_sum = self.alpha * torch.exp ((s_e * p_e))+neg_scores   # [batch_size, n_hops, channel]
        neg_weight = neg_scores/total_sum     # [batch_size, n_hops, channel]
        pos_weight = 1-neg_weight   # [batch_size, n_hops, channel]
        n_e_ = pos_weight * p_e + neg_weight * neg_items_embedding_hardest  # mixing
        return n_e_



    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def loss(self, batch,epoch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_gcn_emb, item_gcn_emb = self.gcn(edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        pos_gcn_embs = item_gcn_emb[pos_item]

        neg_gcn_embs = []
        for k in range(self.K):
            neg_gcn_embs.append(self.negative_sampling(user_gcn_emb, item_gcn_emb,user, neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],pos_item))
        neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user_gcn_emb[user], pos_gcn_embs, neg_gcn_embs)