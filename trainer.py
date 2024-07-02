from dataset import loadTrainData
from model import DenseNet121
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

import torch
import wandb
import json
import os

data_dir="/media/mountHDD3/data_storage/z2h/chestX_ray/data/chest_xray/chest_xray"


def trainer(args):
    # print(torch_mask.shape())
    #set up device
    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    args, train_dl, val_dl, test_dl, inputs, classes = loadTrainData(data_dir, args)

    print(f"#TRAIN batch: {len(train_dl)}")
    print(f"#VAL batch: {len(val_dl)}")
    print(f"#TEST batch: {len(test_dl)}")

    # run_name = get_hash(args)
    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    if args.log:
        run = wandb.init(
        project='seg',
        # entity='truelove',
        config=args,
        name=now,
        force=True
        )

    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{now}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)

    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'

    model = DenseNet121(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_dl) * args.epochs)

    old_valid_loss = 1e26
    
    for epoch in range(args.epochs):
        log_dict = {}

        model.train()
        total_loss = 0 
        total_corrects = 0
        for idx, (inputs, labels) in enumerate(train_dl):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_corrects += torch.sum(preds == labels.data)

        train_mean_loss = total_loss / len(train_dl)
        train_corrects = total_corrects / len(train_dl)

        log_dict['train/loss'] = train_mean_loss 
        log_dict['train/corrects'] = train_corrects 

        print(f"Epoch: {epoch} - Train loss: {train_mean_loss} - Train corrects: {train_corrects}")

        model.eval()    
        with torch.no_grad():
            total_loss = 0
            total_corrects = 0
            for idx, (inputs, labels) in enumerate(val_dl):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_corrects += torch.sum(preds == labels.data)

            valid_mean_loss = total_loss / len(val_dl)
            valid_corrects = total_corrects / len(val_dl)

            log_dict['val/loss'] = valid_mean_loss 
            log_dict['val/corrects'] = valid_corrects 

            print(f"Epoch: {epoch} - Valid loss: {valid_mean_loss} - Valid corrects: {valid_corrects}")

        save_dict = {
            'args' : args,
            'model_state_dict' : model.state_dict()
        }

        if valid_mean_loss < old_valid_loss:
            old_valid_loss = valid_mean_loss

            torch.save(save_dict, best_model_path)
        torch.save(save_dict, last_model_path)  

        if args.log:
            run.log(log_dict)

    if args.log:
        run.log_model(path=best_model_path, name=f'{now}-best-model') 
        run.log_model(path=last_model_path, name=f'{now}-last-model')