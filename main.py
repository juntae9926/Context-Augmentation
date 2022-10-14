import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import numpy as np
from tqdm import tqdm
import argparse

import models
from dataset import PascalVOC_Dataset

from utils import *

def main(args, download_data=False):
    epochs = args.epochs

    # ImageNet Values
    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    """ APPLYing Our Augmentation Method should be here """

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((300, 300)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(330),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    dataset_train = PascalVOC_Dataset(args.data_dir,
                                      year='2012', 
                                      image_set='train', 
                                      download=download_data, 
                                      transform=data_transform['train'], 
                                      target_transform=encode_labels)
    
    dataset_valid = PascalVOC_Dataset(args.data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=download_data, 
                                      transform=data_transform['valid'], 
                                      target_transform=encode_labels)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) 
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = models.resnet50(pretrained=False)
    model = model.to(args.device)
    print(model)

    # Optimizer & Scheduler & Criterion
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD([   
            {'params': list(model.parameters())[:-1], 'lr': args.lr[0], 'momentum': 0.9},
            {'params': list(model.parameters())[-1], 'lr': args.lr[1], 'momentum': 0.9}
            ])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader)), eta_min=1e-5, last_epoch=-1)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    train(args, model, optimizer, scheduler, criterion, train_loader, valid_loader, save_dir=args.save_dir)

def train(args, model, optimizer, scheduler, criterion, train_loader, valid_loader, save_dir):

    tr_loss, tr_map = [], []
    val_loss, val_map = [], []
    best_val_map = 0.0

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format((epoch+1), args.epochs))
        scheduler.step()

        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_ap = 0.0
            m = torch.nn.Sigmoid()

            if phase == 'train': 
                model.train()
                batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
                for x, y in train_loader:
                    x = x.float()
                    x = x.to(args.device)
                    y = y.to(args.device)

                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    batch_bar.set_postfix(loss="{:.04f}".format(loss.item()/args.batch_size), lr="{:.06f}".format(optimizer.param_groups[0]['lr']))

                    running_loss += loss
                    running_ap += get_ap_score(torch.Tensor.cpu(y).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy()) 

                    loss.backward()
                    optimizer.step()
                    
                    del x, y, output
                    torch.cuda.empty_cache()

                    batch_bar.update()
                batch_bar.close()

                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item()/num_samples
                tr_map_ = running_ap/num_samples

                print("Train Loss {:.04f}, Train avg precision: {:.3f}, Learning rate {:.06f}".format(
                    tr_loss_, tr_map_, float(optimizer.param_groups[0]['lr'])))
                
                tr_loss.append(tr_loss_), tr_map.append(tr_map_)
            
            else:
                model.eval()
                batch_bar = tqdm(total=len(valid_loader), dynamic_ncols=True, leave=False, position=0, desc='Valid')
                
                with torch.no_grad():
                    for x, y in valid_loader:
                        x = x.float()
                        x = x.to(args.device)
                        y = y.to(args.device)

                        optimizer.zero_grad()
                        output = model(x)
                        loss = criterion(output, y)
                        batch_bar.set_postfix(loss="{:.04f}".format(loss.item()/args.batch_size), lr="{:.06f}".format(optimizer.param_groups[0]['lr']))

                        running_loss += loss
                        running_ap += get_ap_score(torch.Tensor.cpu(y).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy()) 
                        
                        del x, y, output
                        torch.cuda.empty_cache()

                        batch_bar.update()
                    batch_bar.close()

                    num_samples = float(len(valid_loader.dataset))
                    val_loss_ = running_loss.item()/num_samples
                    val_map_ = running_ap/num_samples

                    val_loss.append(val_loss_), val_map.append(val_map_)
                    print("Valid Loss {:.04f}, Valid avg precision: {:.3f}".format(val_loss_, val_map_))

                    if val_map_ >= best_val_map:
                        best_val_map = val_map_
                        if not os.path.isdir(os.path.join(args.save_dir)):
                            os.mkdir(args.save_dir)
                        torch.save(model.state_dict(), os.path.join(args.save_dir, "model-{}.pth".format(epoch)))
                        print("--- best model saved at {} ---".format(args.save_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="VOCdevkit/VOC2012", type=str)
    parser.add_argument("--save-dir", default="runs", type=str)
    parser.add_argument("--project-name", default="test", type=str)
    parser.add_argument("--device", default="cuda:0", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, type=int, help="Total epochs")
    parser.add_argument("--lr", default=[1.5e-4, 5e-2], type=float, help="Learning rate")
    parser.add_argument("--num-workers", default=8, type=int)


    args = parser.parse_args()
    main(args)