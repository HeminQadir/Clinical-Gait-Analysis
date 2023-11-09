from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import os
import random
import math
import torch

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))



def set_seed(seed=42, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


# Save your trained model.
def save_model(model_folder, outcome_model, epoch): #, imputer, outcome_model, cpc_model):
    torch.save({'model': outcome_model, 'epoch': epoch,}, os.path.join(model_folder, 'model.pt'))

def load_models(model_folder):
    filename = os.path.join(model_folder, 'model.pt')
    state = torch.load(filename)
    model = state['model']
    return model 



#%%
def valid(model, val_loader, local_rank, device):
    # Validation!

    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=local_rank not in [-1, 0])
    
    loss_fct = torch.nn.CrossEntropyLoss()
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for step, data in enumerate(epoch_iterator):

        x = data["data"].type(torch.float32)#torch.tensor(data["data"], dtype=torch.float32)
        y = data["label"].type(dtype=torch.long) #torch.tensor(data["label"], dtype=torch.long)


        x, y = x.to(device), y.to(device)
       
        with torch.no_grad():
            logits = model(x) #[0]   #[0] not needed if we have only one output
            eval_loss = loss_fct(logits, y)

            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            #print(preds)

            predict = y == preds

            TP += (predict == True).sum()
            FP += (predict == False).sum()
            
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    #accuracy = simple_accuracy(all_preds, all_label)
    precision = TP / float(TP+FP) * 100

    print("precision: ", precision)

    return precision #accuracy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self): 
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count