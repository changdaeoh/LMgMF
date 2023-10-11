
import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from time import time,localtime,strftime
from tqdm import tqdm
from copy import deepcopy
#from prettytable import PrettyTable
import pdb
import os.path as osp

from loader import load_data, Loader
import wandb
import argparse

from models import NGCF, DINSMF, LightGCN
from utils import seed_everything
import pdb
import json
from sklearn.metrics import classification_report


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    return feed_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='vanilla')
    parser.add_argument("--wb_name", type=str, default='EDUREC')

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wb_run_name", type=str, default='')
    parser.add_argument("--data_root", type=str, default='/mlainas/DATASET_SHARE/MultimodalEduRec/rating_dataset/')
    parser.add_argument("--iemb_path", type=str, default="/mlainas/DATASET_SHARE/MultimodalEduRec/type2_format1_emb_last.npy")
    parser.add_argument("--uemb_path", type=str, default='/mlainas/DATASET_SHARE/MultimodalEduRec/type1_format1_userst2_emb_last.npy')
    parser.add_argument("--data_prop", type=str, default='full', choices = ['full','010','030','050','070','090'])
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--grade", type=str, default='1', help='target grade')
    parser.add_argument("--dataset", type=str, default='benchmark')

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=1024)

    # mf / lmgmf
    parser.add_argument("--ld", type=int, default=500)
    parser.add_argument("--reg_b", type=float, default=0.0)
    parser.add_argument("--reg_f", type=float, default=0.0)
    parser.add_argument("--dist_u", type=float, default=1e-3)
    parser.add_argument("--dist_i", type=float, default=1e-2)
    parser.add_argument("--prompt_type", type=int, default=1)

    # ncf
    parser.add_argument("--ncf_layer", type=int, default=3)

    # emb clf
    parser.add_argument("--clf_type", type=str, default='mlp')

    # vfm
    parser.add_argument("--kl_coef", type=float, default=1.0)

    # dins
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--mess_dropout", type=bool, default=False)
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1)
    parser.add_argument("--edge_dropout", type=bool, default=False)
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1)
    parser.add_argument("--ns", type=str, default='mixgcf', help="rns,dins")
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")
    parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")
    parser.add_argument("--alpha", type=float, default='0.5', help="Control the trend of the model to positive contributation")
    parser.add_argument("--context_hops", type=int, default=3, help="hop")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60]',
                        help='Output sizes of every layer')

    n_users = 0
    n_items = 0
    """read args"""
    global args, device
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.wb_name:
        wandb.init(project=args.wb_name, name=args.wb_run_name)
        wandb.config.update(args)

    print(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu}")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    #! original dataset
    tr_ds = np.load(args.data_root + f'/rating_train_seed{args.seed}.npy')
    te_ds = np.load(args.data_root + f'/rating_test_seed{args.seed}.npy')
    # n_users = int(len(np.unique(tr_ds[:,0])))
    # n_items = int(len(np.unique(np.concatenate((tr_ds[:,1], te_ds[:,1]),axis=0))))
    # pdb.set_trace()
    test_loader = Loader(te_ds[:,:2], te_ds[:,2], batchsize=len(te_ds))
    # n_params['n_users'] = n_users
    # n_params['n_items'] = n_items
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    
    # if args.gnn == 'lightgcn':
    #     model = LightGCN(n_params, args, norm_mat).to(device)
    # elif args.gnn == "mf":
    #     model = Matrix_Factorization(n_params, args).to(device)
    # else:
    if args.method == 'dins-ngcf':
        model = NGCF(n_params, args, norm_mat).to(device)
    elif args.method == 'dins-mf':
        model = DINSMF(n_params, args).to(device)
    elif args.method == 'dins-lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        raise ValueError


    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    best_acc = 0
    should_stop = False
    best_pred = None

    print("start training ...")
    for epoch in tqdm(range(args.epochs)):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        while s + args.bs <= len(train_cf):
            batch = get_feed_dict(train_cf_ , user_dict['train_user_set'], s , s + args.bs, n_negs)
            batch_loss,bpr_loss,reg_loss= model.loss(batch, epoch)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.bs
        
        model.eval()
        user_gcn_emb, item_gcn_emb = model.generate()
        # print(f'ep{epoch} u emb ele0',user_gcn_emb[0,0])
        # print(f'ep{epoch} i emb ele0',item_gcn_emb[0,0])
        with torch.no_grad():
            for i, (ids, y) in enumerate(test_loader):
                uid, iid = ids[:,0].to(f'cuda:{args.gpu}'), ids[:,1].to(f'cuda:{args.gpu}')
                u_g_embeddings = user_gcn_emb[uid]
                i_g_embddings  = item_gcn_emb[iid]
                pred = np.array(F.sigmoid(model(u_g_embeddings, i_g_embddings).diag().detach().cpu()) > 0.5, dtype=np.float32)

                if i == 0:
                    tot_pred = pred
                else:
                    tot_pred = np.concatenate((tot_pred, pred), axis=0)
        
        #pdb.set_trace()
        tot_pred = tot_pred.astype(np.int32)

        acc = (te_ds[:,2] == tot_pred).mean()
        if acc > best_acc:
            out = classification_report(te_ds[:,2], tot_pred, labels=[0,1], output_dict=True)['1']
            prec = out['precision']
            rec = out['recall']
            f1 = out['f1-score']
            # prec = 0
            # rec = 0
            # f1 = 0
            best_acc = acc
            wandb.log({'best_ep':epoch,'best_acc':acc,'best_prec':prec,'best_rec':rec,'best_f1':f1})
            best_pred = tot_pred
            
            print({'best_ep':epoch,'best_acc':acc,'best_prec':prec,'best_rec':rec,'best_f1':f1})

        print('training loss at epoch %d: %.4f' % (epoch, loss.item()))

    best_pred = pd.DataFrame(best_pred)
    os.makedirs(f'/mlainas/WEIGHT_SHARE/MultimodalEduRec/bastmodels/{args.method}/{args.wb_run_name}/seed{args.seed}/', exist_ok=True)
    best_pred.to_csv(f'/mlainas/WEIGHT_SHARE/MultimodalEduRec/bastmodels/{args.method}/{args.wb_run_name}/seed{args.seed}/entire.csv')