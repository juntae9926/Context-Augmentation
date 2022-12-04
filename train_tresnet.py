import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix

import os
import numpy as np
from tqdm import tqdm
import argparse

from dataset import PascalVOC_Dataset, CustomDataset, my_collate
import wandb

from math import ceil
from utils import *
from schedulers import CosineAnnealingWarmUpRestarts

from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from src_files.helper_functions.helper_functions import add_weight_decay, CutoutPIL
from randaugment import RandAugment

import warnings
warnings.filterwarnings(action='ignore')

def compute_mAP(targs, preds, class_ap=False):
    targs = targs.cpu().numpy()
    preds = preds.cpu().numpy()

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = compute_AP(scores, targets)
    if class_ap:
        return ap
    else:
        return ap.mean()


def compute_AP(output, label):
    epsilon = 1e-8
    
    indices = output.argsort()[::-1]
    total_count_ = np.cumsum(np.ones((len(output), 1)))
    
    label_ = label[indices]
    ind = label_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def set_transforms():
    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    data_transform = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop((300, 300)),
            # transforms.RandomVerticalFlip(),
            transforms.Resize((args.image_size, args.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            # transforms.Resize(330),
            # transforms.CenterCrop(300),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    return data_transform


# def load_model(num_classes, pretrained=True):
#     model = models.resnet101(pretrained=pretrained)
#     in_ftrs = model.fc.in_features
#     model.fc = nn.Linear(in_ftrs, num_classes)

#     return model

def load_model(args):
    model = create_model(args)

    return model


def train(args, model, optimizer, scheduler, criterion, train_loader):
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    total_loss = 0
    mAP = []
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for x, y in train_loader:
        x, y = x.to(args.device), y.to(args.device)

        
        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = criterion(pred, y)

        total_loss += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        scheduler.step()

        running_map = compute_mAP(y.data, pred.data)
        mAP.append(running_map)

        if args.wandb:
            wandb.log({"learning rate": optimizer.param_groups[0]['lr']})

        batch_bar.set_postfix(loss="{:.04f}".format(loss.item()), mAP="{:.04f}".format(running_map), lr="{:.06f}".format(optimizer.param_groups[0]['lr']))

        del x, y, pred
        torch.cuda.empty_cache()

        batch_bar.update()
    batch_bar.close()

    epoch_loss = total_loss / ceil(len(train_loader.dataset)/train_loader.batch_size)
    mAP_mean = sum(mAP) / len(mAP)

    if args.wandb:
        wandb.log({"Train loss": epoch_loss})
        wandb.log({"Train mAP": mAP_mean})
        
    return epoch_loss, mAP_mean


def valid(args, model, criterion, valid_loader):
    
    model.eval()
    total_loss = 0
    mAP = []
   
    batch_bar = tqdm(total=len(valid_loader), dynamic_ncols=True, leave=False, position=0, desc='Valid')
    for x, y in valid_loader:
        x, y = x.to(args.device), y.to(args.device)

        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss

            running_map = compute_mAP(y.data, pred.data)
            mAP.append(running_map)

            batch_bar.set_postfix(loss="{:.04f}".format(loss.item()), mAP="{:.04f}".format(running_map))

            del x, y, pred
            torch.cuda.empty_cache()

            batch_bar.update()
        batch_bar.close()

    epoch_loss = total_loss / ceil(len(valid_loader.dataset)/valid_loader.batch_size)
    mAP_mean = sum(mAP) / len(mAP)

    if args.wandb:
        wandb.log({"Valid loss": epoch_loss})
        wandb.log({"Valid mAP": mAP_mean})

    return  epoch_loss, mAP_mean

def test(args, test_loader, save_dir=None):

    model = load_model(num_classes=80, pretrained=False).to(args.device)
    if save_dir:
        model.load_state_dict(torch.load(os.path.join(save_dir, "best.pth")))
    else:
        model.load_state_dict(torch.load(args.test_model))
    model.eval()

    torch.cuda.empty_cache()

    APs = np.zeros((80))
    for x, y in tqdm(test_loader):
        x, y = x.to(args.device), y.to(args.device)

        with torch.no_grad():
            pred = model(x)

            running_ap = compute_mAP(y.data, pred.data, class_ap=True)
            APs = np.vstack((APs, running_ap))
        
            del x, y, pred
            torch.cuda.empty_cache()
    class_ap = 100 * APs.sum(axis=0) / (APs.shape[0] -1)
    mAP = class_ap.mean()
    print("mAP is {:0.2f} \nClass_aps are {}".format(mAP, np.round(class_ap, 2)))



def main(args):

    if not os.path.isdir(args.save_dir):
        save_dir = os.path.join(args.save_dir, 'test_0')
        os.makedirs(save_dir)
    else:
        weight_dirs = os.listdir(args.save_dir)
        save_dir = os.path.join(args.save_dir, "test_%d" % len(weight_dirs))
        os.makedirs(save_dir)

    data_transforms = set_transforms()

    dataset_train = CustomDataset(root=args.data_dir, partition="train2017", use_method = True, annFile=os.path.join(args.data_dir, "annotations/instances_train2017.json"), transforms=data_transforms["train"], k=args.k)
    dataset_valid = CustomDataset(root=args.data_dir, partition="val2017", use_method = False, annFile=os.path.join(args.data_dir, "annotations/instances_val2017.json"), transforms=data_transforms["valid"])
    print("len(train_dataset)): ", len(dataset_train), "len(val_dataset)): ", len(dataset_valid))

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,  shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # Model
    model = load_model(args)
    model = model.to(args.device)

    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs, pct_start=0.2)

    log_file = open(os.path.join(save_dir, "train_log.txt"), "w")
    log_file.write(f"--- Experiment with method {args.use_method} --- \n")

    best_map = 0
    for epoch in range(args.epochs):
        print("Epoch {}/{}".format((epoch+1), args.epochs))
        log_file = open(os.path.join(save_dir, "train_log.txt"), "w")
        log_file.write("Epoch {}/{} \n".format((epoch+1), args.epochs))

        # Training
        train_loss, train_map = train(args, model, optimizer, scheduler, criterion, train_loader)
        print("Train Loss {:.04f}, mAP {:.04f}, Learning rate {:.06f}".format(train_loss, train_map, float(optimizer.param_groups[0]['lr'])))
        log_file = open(os.path.join(save_dir, "train_log.txt"), "w")
        log_file.write("Train Loss {:.04f}, mAP {:.04f}, Learning rate {:.06f} \n".format(train_loss, train_map, float(optimizer.param_groups[0]['lr'])))

        # Validation
        valid_loss, valid_map = valid(args, model, criterion, valid_loader)
        print("Valid Loss {:.04f}, mAP {:.4f}".format(valid_loss, valid_map))
        log_file = open(os.path.join(save_dir, "train_log.txt"), "w")
        log_file.write("Valid Loss {:.04f}, mAP {:.4f} \n".format(valid_loss, valid_map))


        # Early stopping
        if best_map != 0 and 0 < best_map-valid_map < 0.00001:
            print(f"--- early stopped at epoch {epoch+1} ---")
            break

        # save best weight
        if valid_map > best_map:
            best_map = valid_map
            
            torch.save(model.state_dict(), os.path.join(save_dir, "epoch-{}.pth".format(epoch+1)))
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            print("--- best model saved at {} ---".format(save_dir))

            weight_files = os.listdir(save_dir).sort()
            if weight_files and len(weight_files) > 10:
                os.remove(os.path.join(args.save_dir, weight_files[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--save-dir", default="runs/pretrained/method", type=str)
    parser.add_argument("--project-name", default="base", type=str)
    parser.add_argument("--device", default="cuda:0", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument("--epochs", default=40, type=int, help="Total epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--k", default=1, type=int, help="set maximum pairs to use the method")
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="", type=str)
    parser.add_argument("--use-method", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-model", default="./runs/pretrained/no_method/test_0/best.pth", type=str, help="set test model path")
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
    parser.add_argument('--num-classes', default=80)
    parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)
    
    
    args = parser.parse_args()
    print(args)

    if args.wandb == True:    
        wandb.init(project=args.project_name, entity=args.wandb_entity)
        wandb.config.update(args)
        print(f"Start with wandb with {args.project_name} || method=={args.use_method}")
    else:
        print(f"Start without wandb || method=={args.use_method}")

    args = parser.parse_args()
    main(args)