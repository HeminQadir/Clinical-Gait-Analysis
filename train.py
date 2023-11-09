from data_loader import *
from utils import *
from models import Classification_1DCNN
from tqdm import tqdm
import torch
from config import get_config
import sys 


def train(config, data_folder, model_folder, device):

   
    t_total = config.num_steps

    split = True
    split_ratio = 0.1
    shuffle = True

    if split:
        X_train, X_val = load_train_val_files(data_folder, split, split_ratio)
        
        trainset = dataset(data_folder, X_train)

        train_loader = DataLoader(trainset, batch_size=config.train_batch_size,shuffle=shuffle)

        valset = dataset(data_folder, X_val, device)
        val_loader =  DataLoader(valset, batch_size=config.eval_batch_size, shuffle=shuffle)

    else:
        X_train = load_train_val_files(data_folder, split, split_ratio)
        trainset = dataset(data_folder, X_train)
        train_loader =  DataLoader(trainset, batch_size=config.train_batch_size, shuffle=shuffle)

    model = Classification_1DCNN(config)
    model = model.to(device)

    weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config.learning_rate,
                                betas=(0.9, 0.999), 
                                eps=1e-08,
                                weight_decay=weight_decay)

    
    if config.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=config.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config.warmup_steps, t_total=t_total)

    model.zero_grad()

    seed = 42
    set_seed(seed=seed, n_gpu=1)

    losses = AverageMeter()

    global_step, best_acc = 0, 0

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=config.local_rank not in [-1, 0])

        for step, data in enumerate(epoch_iterator):

            x = data["data"].type(torch.float32)#torch.tensor(data["data"], dtype=torch.float32)
            y = data["label"].type(dtype=torch.long) #torch.tensor(data["label"], dtype=torch.long)

            x, y = x.to(device), y.to(device)

            #print("I am label: ", y)
            loss = model(x, y) 

            loss.backward()

            if (step + 1) % config.gradient_steps == 0:
                losses.update(loss.item()*config.gradient_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                if global_step % config.eval_every == 0 and config.local_rank in [-1, 0]:
                    accuracy = valid(model, val_loader, config.local_rank, device)
                    if best_acc < accuracy:
                        #save_challenge_model(args, model)
                        save_model(model_folder, model, global_step)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break


if __name__ == '__main__':

    # Parse the arguments.
    if not (len(sys.argv) == 3):
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    data_folder = sys.argv[1]       #data_folder = "/home/jacobo/Eirik/dataset"
    model_folder = sys.argv[2]      #model_folder = "/home/jacobo/Eirik/models"

    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(config, data_folder, model_folder, device)


# RUN: 
# python train.py /home/jacobo/Eirik/dataset /home/jacobo/Eirik/models