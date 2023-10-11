'''
    borrowed from https://github.com/khanhnamle1994/MetaRec
'''

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.engine.engine import Engine
from ignite.metrics import Loss, Accuracy
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.contrib.metrics import ROC_AUC
import pdb

from tensorboardX import SummaryWriter

from loader import Loader, load_data
from datetime import datetime

from models import *
from utils import output_transform, seed_everything, custom_trainer, custom_evaluator
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--wb_run_name", type=str, default='')
parser.add_argument("--data_root", type=str, default='./data/iscream/rating_dataset/')
parser.add_argument("--iemb_path", type=str, default="./data/iscream/type2_format1_emb_last.npy")
parser.add_argument("--uemb_path", type=str, default='./data/iscream/type1_format1_userst2_emb_last.npy')
parser.add_argument("--data_prop", type=str, default='full', choices = ['full','010','030','050','070','090'])
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--grade", type=str, default='1', help='target grade')
parser.add_argument("--dataset", type=str, default='iscream', choices = ['iscream','benchmark'])

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--bs", type=int, default=1024)

parser.add_argument("--ld", type=int, default=500)
parser.add_argument("--reg_b", type=float, default=0.0)
parser.add_argument("--reg_f", type=float, default=0.0)
parser.add_argument("--dist_u", type=float, default=1e-3)
parser.add_argument("--dist_i", type=float, default=1e-2)
parser.add_argument("--prompt_type", type=int, default=1)

parser.add_argument("--ncf_layer", type=int, default=3)

parser.add_argument("--clf_type", type=str, default='mlp')

parser.add_argument("--method", type=str, default='vanilla')
parser.add_argument("--wb_name", type=str, default='EDUREC')

parser.add_argument("--kl_coef", type=float, default=1.0)

args = parser.parse_args()

seed_everything(args.seed)
if args.wb_name:
    wandb.init(project=args.wb_name, name=args.wb_run_name)
    wandb.config.update(args)

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

# Load preprocessed data
grade_prefix = 'entire' if args.grade == 'entire' else f'grade{args.grade}'
if args.dataset == 'benchmark':
    print(f'load dataset rating_train/test_seed{args.seed}.npy')
    fh_tr = np.load(args.data_root + f'/{args.data_prop}/rating_train_seed{args.seed}.npy')
    fh_te = np.load(args.data_root + f'/{args.data_prop}/rating_test_seed{args.seed}.npy')
else:
    print(f'load dataset rating_train/test_{grade_prefix}_seed{args.seed}.npy')
    fh_tr = np.load(args.data_root + f'/{args.data_prop}/rating_train_{grade_prefix}_seed{args.seed}.npy')
    fh_te = np.load(args.data_root + f'/{args.data_prop}/rating_test_{grade_prefix}_seed{args.seed}.npy')
                                   
# Note Pytorch is finicky about need int64 types
train_x = fh_tr[:,:2].astype(np.int64)
#train_x[:,0] = np.array(range(len(np.unique(train_x[:,0]))))
train_y = fh_tr[:,2]

# We've already split the data into train & test set
test_x = fh_te[:,:2].astype(np.int64)
#test_x[:,0] = np.array(range(len(np.unique(test_x[:,0]))))
test_y = fh_te[:,2]

# Extract the number of users and number of items
n_user = int(len(np.unique(train_x[:,0])))
n_item = int(len(np.unique(np.concatenate((train_x[:,1], test_x[:,1]),axis=0))))


# Define the Hyper-parameters
model_args = {'ld':args.ld,
              'reg_bias':args.reg_b,
              'reg_feat':args.reg_f,
              'dist_i':args.dist_i,
              'dist_u':args.dist_u,
              'grade':args.grade} #! add grade for g-wise prediction

# Setup logging
os.makedirs(f'log/runs_{args.dataset}/{args.method}/{args.wb_run_name}/seed{args.seed}/', exist_ok=True)
log_dir = f'log/runs_{args.dataset}/{args.method}/{args.wb_run_name}/seed{args.seed}/{grade_prefix}_bs{args.bs}_lr{args.lr}_ld{args.ld}_rb{args.reg_b}_rf{args.reg_f}_du{args.dist_u}_di{args.dist_i}_p{args.prompt_type}'
writer = SummaryWriter(log_dir=log_dir)

# Instantiate the model class object
print(f'build model with param:\n',model_args)

lmclfs = ['lmlr_u','lmlr_i','lmlr_ui','lmmlp_u','lmmlp_i','lmmlp_ui']
lminits = ['lminit_u','lminit_i','lminit_ui']
lmdistills = ['lmdistill_u','lmdistill_i','lmdistill_ui']
if args.method in lminits:       initpath = (args.uemb_path, args.iemb_path)
elif args.method in lmdistills:  initpath = (args.uemb_path, args.iemb_path)
elif args.method in lmclfs:      initpath = (args.uemb_path, args.iemb_path)
else:                            initpath = 'noinit'

if args.method.split('_')[0] in ['lmlr','lmmlp']:
    model = EmbClassifier(writer, device, initpath, args.method, args.clf_type).to(device)
elif 'vfm' in args.method:
    model = VFM(train_x, test_x, args.ld, device, args.kl_coef, output='class').to(device)
elif args.method == 'ncf':
    md = 'NeuMF-end'
    model = NCF(n_user, n_item, args.ld, args.ncf_layer, 0.0, md, device).to(device)
# elif args.method == 'dins-ngcf':
#     model = NGCF(n_params, args, norm_mat).to(device)
else:
    model = MF(n_user, n_item, writer, device, initpath, args.method, args.dataset, **model_args).to(device)

# Use Adam optimizer
print(f'optimization configs: epochs={args.epochs}, lr={args.lr}, bs={args.bs}',)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Create a supervised trainer
if args.method in lmclfs + lminits + lmdistills + ['vanilla','ncf']:
    trainer = create_supervised_trainer(model, optimizer, model.loss)
else:
    trainer = custom_trainer(model, optimizer, model.loss, device, args.method)
    

# Create a metrics dictionary
m_recall = Recall(output_transform=output_transform)
m_precision = Precision(output_transform=output_transform)
m_auc = ROC_AUC(output_transform=output_transform)

metrics = {'accuracy': Accuracy(output_transform=output_transform), 
           'recall': m_recall,
           'precision': m_precision,
           'f1': (m_precision * m_recall * 2 / (m_precision + m_recall)).mean(),
           'auc':m_auc}

# Create a supervised evaluator
if args.method in lmclfs + lminits + lmdistills + ['vanilla','ncf']:
    evaluator = create_supervised_evaluator(model, metrics=metrics)
else:
    evaluator = custom_evaluator(model, metrics=metrics, device=device, method=args.method)

# Load the train and test data
train_loader = Loader(train_x, train_y, batchsize=args.bs)
test_loader = Loader(test_x, test_y, batchsize=args.bs)


def log_training_loss(engine, log_interval=args.log_interval):
    """
    Function to log the training loss
    """
    model.itr = engine.state.iteration  # Keep track of iterations
    if model.itr % log_interval == 0:
        fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"
        # Keep track of epochs and outputs
        # msg = fmt.format(engine.state.epoch, engine.state.iteration, len(train_loader)*args.epochs, engine.state.output)
        # print(msg)


trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)

#best_acc=0; best_prec=0; best_rec=0; best_f1=0; best_ep=0
def log_validation_results(engine):
    """
    Function to log the validation performance
    """
    # When triggered, run the validation set
    
    evaluator.run(test_loader)
    # Keep track of the evaluation metrics
    avg_acc = evaluator.state.metrics['accuracy']
    avg_prec = evaluator.state.metrics['precision']
    avg_rec = evaluator.state.metrics['recall']
    avg_f1 = evaluator.state.metrics['f1']
    avg_auc = evaluator.state.metrics['auc']
    print("Epoch[{}] Val Acc: {:.3f}, Prec: {:.3f}, Rec: {:.3f}, F1: {:.3f}, AUC: {:.3f} ".format(engine.state.epoch, avg_acc, avg_prec, avg_rec, avg_f1, avg_auc))
    writer.add_scalar("validation/avg_acc", avg_acc, engine.state.epoch)
    writer.add_scalar("validation/avg_prec", avg_prec, engine.state.epoch)
    writer.add_scalar("validation/avg_rec", avg_rec, engine.state.epoch)
    writer.add_scalar("validation/avg_f1", avg_f1, engine.state.epoch)
    writer.add_scalar("validation/avg_auc", avg_auc, engine.state.epoch)

    if engine.state.epoch == 1:
        engine.state.best_acc = 0.0
        engine.state.best_prec = 0.0
        engine.state.best_rec = 0.0
        engine.state.best_f1 = 0.0
        engine.state.best_auc = 0.0
        engine.state.best_ep = 1

    if avg_acc > engine.state.best_acc:
        engine.state.best_acc = avg_acc
        engine.state.best_prec = avg_prec
        engine.state.best_rec = avg_rec
        engine.state.best_f1 = avg_f1
        engine.state.best_auc = avg_auc
        engine.state.best_ep = engine.state.epoch

        # Save best model
        os.makedirs(f'bestmodels/{args.method}/{args.wb_run_name}/seed{args.seed}/', exist_ok=True)
        torch.save(model.state_dict(), f'bestmodels/{args.method}/{args.wb_run_name}/seed{args.seed}/{grade_prefix}.pth')
        
        # Save best model prediction output
        pred = model(torch.from_numpy(test_x))
        if 'vfm' in args.method:
            pred = torch.tensor(F.sigmoid(pred[0].mean.squeeze()) > 0.5, dtype=torch.float32)
        else:
            pred = torch.tensor(F.sigmoid(pred) > 0.5, dtype=torch.float32)
        #pdb.set_trace()
        pred_pd = pd.DataFrame((pred.detach().cpu().numpy()).astype(int))
        pred_pd.to_csv(f'bestmodels/{args.method}/{args.wb_run_name}/seed{args.seed}/{grade_prefix}.csv')

    wandb.log({'val_acc':avg_acc,'val_prec':avg_prec,'val_rec':avg_rec,'val_f1':avg_f1})

    if engine.state.epoch == args.epochs:
        wandb.log({'best_ep':engine.state.best_ep,'best_acc':engine.state.best_acc,
                    'best_prec':engine.state.best_prec,'best_rec':engine.state.best_rec,'best_f1':engine.state.best_f1,'best_auc':engine.state.best_auc,
                    'train_y':train_y.mean(), 'test_y':test_y.mean()})
        print(f"best ep: {engine.state.best_ep}, acc:{engine.state.best_acc:.3f}, prec:{engine.state.best_prec:.3f}, rec:{engine.state.best_rec:.3f}, f1:{engine.state.best_f1:.3f}, yprop(tr/te):({train_y.mean():.2f}/{test_y.mean():.2f})")


trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

# Train the model
print(f'start train {args.epochs}epochs')
trainer.run(train_loader, max_epochs=args.epochs)