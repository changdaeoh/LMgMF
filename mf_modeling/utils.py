import torch
import torch.nn.functional as F
import random
import numpy as np
import os
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.metrics import MetricsLambda

def output_transform(output):
    y_pred, y = output
    y_pred = torch.tensor(F.sigmoid(y_pred) > 0.5, dtype=torch.float32)
    return y_pred, y

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def custom_trainer(model, optimizer, loss_fn, device, method):
    def _update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if 'vfm' in method:
            ll, kl_reg = model(x)
            loss = loss_fn(ll, kl_reg, y)
        else:
            raise ValueError(f'{method} method is not implemented yet')

        # Backpropagation
        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item()
        }

    trainer = Engine(_update_fn)

    # Add RunningAverage to calculate and print the average loss during training
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'avg_loss')

    return trainer


def custom_evaluator(model, metrics=None, device=None, non_blocking=False, method='vfm'):
    if device is None:
        device = model.device

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Forward pass through the model
            #! modifier (only take first element from model output)
            if 'vfm' in method:
                # from distribution to scalar
                output = model(x)[0].mean.squeeze() #.detach()
            else:
                raise ValueError(f'{method} is not implemented yet')

            return output, y

    evaluator = Engine(_inference)

    # def parser(x):
    #     try:
    #         n = len(x)
    #     except:
    #         n = -1
        
    #     if n == 2:
    #         return x[0]
    #     else:
    #         return x

    if metrics:
        for name, metric in metrics.items():
            metric_output_transform = lambda x: x
            MetricsLambda(metric_output_transform, metric).attach(evaluator, name)

    return evaluator