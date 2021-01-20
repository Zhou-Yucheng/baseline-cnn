#!/usr/bin/python3.7

import os, csv, socket, time, shutil, pathlib
from datetime import datetime

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from torch.nn.modules.loss import TripletMarginWithDistanceLoss
import torch.optim as optim
import torchvision
from sklearn import manifold
from torch.utils.tensorboard import SummaryWriter

import data


def plot_model_history(log_path, plot_loss=True):
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    with open(log_path) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            if (not row) or row[0] == 'Epoch':
                continue
            if '#' in row[0]:
                break
            row = list(map(float, row))
            train_loss.append(row[1])
            train_acc.append(row[2])
            val_loss.append(row[3])
            val_acc.append(row[4])

    plt.plot(train_acc, linewidth=0.5)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    if plot_loss:
        plt.figure()
        plt.plot(train_loss, linewidth=0.5)
        plt.plot(val_loss)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()


def plot_model_CM(model, valid_loader):
    model.eval()
    CM = torch.zeros((n_classes, n_classes))
    n_correct = 0
    n_valid = 0
    epoch_valid = 9

    with torch.no_grad():
        for ep in range(epoch_valid):
            for inputs, labels in valid_loader:
                labels = labels.to(device)

                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                n_correct += torch.sum(preds == labels.data)
                n_valid += labels.shape[0]
                for i in range(labels.shape[0]):
                    CM[preds[i]][labels[i]] += 1

    df_cm = pd.DataFrame(CM.numpy(), range(n_classes), range(n_classes))
    # plt.figure(figsize=(10,7))
    # sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap='Greys')  # font size
    plt.show()
    log('\nThe plotted confusion matrix is based on {} random valid sample'.format(n_valid))
    log('\tValid Acc = {:4f} ({}/{})'.format((n_correct + 0.0) / n_valid, n_correct, n_valid))


def plot_tsne(model, valid_loader):
    model.eval()
    epoch_plot = 9
    time_start = time.time()

    x, y = None, None
    with torch.no_grad():
        for ep in range(epoch_plot):
            for inputs, labels in valid_loader:
                labels = labels.to(device)

                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                x = torch.cat((x, model.feature_map),
                              0) if x is not None else model.feature_map
                y = torch.cat((y, preds), 0) if y is not None else preds
        x = x.squeeze().cpu().numpy()
        y = y.cpu().numpy()

    # log('Time for retrieving feature map:', time.time() - time_start)

    # x = np.load('x.npy')
    # y = np.load('y.npy')

    def plot_embedding(x_, title=None):
        x_min, x_max = np.min(x_, 0), np.max(x_, 0)
        x_ = (x_ - x_min) / (x_max - x_min)

        plt.figure()
        for i in range(x_.shape[0]):
            plt.text(x_[i, 0], x_[i, 1], str(y[i]), color=plt.cm.tab20(y[i]), fontdict={'weight': 'bold', 'size': 9})

        if title is not None:
            plt.title(title)
        plt.show()
        # plt.savefig('../photos/{} t-SNE features visualization')

    tsne = manifold.TSNE(n_components=2, init='pca')
    x_tsne = tsne.fit_transform(x)
    plot_embedding(x_tsne)
    log('Time for TSNE:', time.time() - time_start)


def vis_conv(model, valid_loader):
    model.eval()
    time_start = time.time()
    # log('Start')

    x0, x1, x2, x3, x4, x, y = None, None, None, None, None, None, None
    with torch.no_grad():
        for inputs, labels in valid_loader:
            labels = labels.to(device)

            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            x0 = torch.cat((x0, model.x0), 0) if x0 is not None else model.x0
            x1 = torch.cat((x1, model.x1), 0) if x1 is not None else model.x1
            x2 = torch.cat((x2, model.x2), 0) if x2 is not None else model.x2
            x3 = torch.cat((x3, model.x3), 0) if x3 is not None else model.x3
            x4 = torch.cat((x4, model.x4), 0) if x4 is not None else model.x4

            x = torch.cat((x, model.feature_map), 0) if x is not None else model.feature_map
            y = torch.cat((y, preds), 0) if y is not None else preds
            # break

        x0 = x0.cpu().numpy()
        x1 = x1.cpu().numpy()
        x2 = x2.cpu().numpy()
        x3 = x3.cpu().numpy()
        x4 = x4.cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.cpu().numpy()
        # log('Time for retrieving feature map:', time.time() - time_start)

    def plot_conv_feature_map(x_, i_b, n_square):
        plt.figure()
        n = 1
        for _ in range(n_square):
            for _ in range(n_square):
                ax = plt.subplot(n_square, n_square, n)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(x_[i_b, n - 1, :, :], cmap='Greys')
                n += 1
        plt.show()

    i_batch = 1
    n_square = 8
    plot_conv_feature_map(x0, i_batch, n_square)
    plot_conv_feature_map(x1, i_batch, n_square)
    plot_conv_feature_map(x2, i_batch, n_square)
    plot_conv_feature_map(x3, i_batch, n_square)
    plot_conv_feature_map(x4, i_batch, n_square)
    log('Time for visualize conv layer:', time.time() - time_start)


def train():
    model.train(True)
    total_loss = 0.0
    total_correct = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)

    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = total_correct.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc.item()


def evaluate():
    model.train(False)
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()


def log_and_save():
    global best_val_acc, best_epoch
    lr = optimizer.param_groups[0]['lr']
    log(f"epoch: {epoch}  lr:{lr}  train_loss: {train_loss:.3f}  val_loss: {val_loss:.3f}  train_acc: {train_acc:.3f}  val_acc: {val_acc:.3f}")

    # ===Save
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), f'./models/{args.model_name}_final.pth')

    # ===Writer
    fwriter.write(f'{epoch}, {train_loss}, {train_acc}, {val_loss}, {val_acc}, {lr}\n')
    swriter.add_scalar('Train/Loss', train_loss, epoch)
    swriter.add_scalar('Train/Acc', train_acc, epoch)
    swriter.add_scalar('Valid/Loss', val_loss, epoch)
    swriter.add_scalar('Valid/Acc', val_acc, epoch)

    # ===Print result, Rename (copy) saved model, Close writer
    if epoch == args.epochs:
        log(f'Best Valid Acc: {best_val_acc:.4f}, at epoch: {best_epoch}')
        shutil.copy(f'./models/{args.model_name}_final.pth', f'./models/{model_fullname}-ep{best_epoch}-acc{best_val_acc*100:.1f}.pth')
        fwriter.close()
        swriter.close()


def get_model(model_name, n_classes):
    model = eval(f'torchvision.models.{model_name}(pretrained=True)')
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)
    
    return model


def lr_warmup_step_schedule(epoch):
    if epoch < args.warmup_epochs:
        return epoch/args.warmup_epochs
    elif epoch < args.step:
        return 1.0
    elif args.step <= epoch < args.step2:
        return args.gamma
    else: # args.step2 <= epoch
        return args.gamma**2


def get_args():
    parser = argparse.ArgumentParser(description='Baseline CNN')

    parser.add_argument('-m', '--model_name', type=str, default='resnet18', help='model name in torchvision')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='num of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-c', '--cuda', type=str, default='', help='cuda visible device id')
    parser.add_argument('-r', '--resume', action='store_true', help='resume training')
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help='lr gamma')
    parser.add_argument('-w', '--warmup_epochs', type=int, default=5, help='lr warmup epochs')
    parser.add_argument('-s', '--step', type=int, default=1000, help='lr decrease step')
    parser.add_argument('--step2', type=int, default=2000, help='lr decrease step')
    parser.add_argument('--eval_only', action='store_true', help='eval model and exit')
    parser.add_argument('--fp16', type=int, default=1, help="FP16 acceleration, use 0/1 for false/true")
    # Requires pytorch>=1.6 to use native fp 16 acceleration (https://pytorch.org/docs/stable/notes/amp_examples.html)

    args_ = parser.parse_args()

    args_.fp16 = bool(args_.fp16)
    host_name = socket.gethostname().lower()
    if not args_.cuda:
        args_.cuda = '1' if host_name == 'dell-poweredge-t640' else '0'

    return args_


def log(msg, end='\n'):
    log_path='./logs/log.log'
    dt_now=datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    with open(log_path, 'a+') as fp:
        fp.write(f'[{dt_now}] {msg}{end}')
    print(f'[{dt_now}] {msg}', end=end)


if __name__ == '__main__':
    args = get_args()
    start_time = time.time()
    n_classes = 20
    input_size = 224

    train_loader, valid_loader = data.get_image_folder_data_loader(
        data_dir='../data/ClipArt20', input_size=input_size, batch_size=args.batch_size)
    
    model = get_model(args.model_name, n_classes)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda")
    model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, args.step, args.gamma)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_warmup_step_schedule)
    criterion = nn.CrossEntropyLoss()
    _lr = f'{args.lr * 1000}x'  # lr 1x = 1/1000
    _r = '-r' if args.resume else ''
    _s = 's' if args.step < args.epochs else ''
    model_fullname = f"{args.model_name}-bs{args.batch_size:02d}-lr{_s}{_lr}{_r}".replace('e-0', 'e-')

    if args.eval_only:
        model.load_state_dict(torch.load(f'./models/{args.model_name}_final.pth'))
        plot_model_history(f'./logs/{model_fullname}.csv')
        plot_tsne(model, valid_loader)
        vis_conv(model, valid_loader)
        plot_model_CM(model, valid_loader)
    if args.resume:
        model.load_state_dict(torch.load(f'./models/{args.model_name}_final.pth'))
    
    pathlib.Path('./logs/csvs').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./logs/runs').mkdir(exist_ok=True)
    pathlib.Path('./models').mkdir(exist_ok=True)
    dt_now = datetime.now().strftime('%m%d-%H%M')
    swriter = SummaryWriter(log_dir=f'./logs/runs/{dt_now} {model_fullname}')
    fwriter = open(f'./logs/csvs/{dt_now} {model_fullname}.csv', 'a+')
    fwriter.write('Epoch, Train Loss, Train Acc, Valid Loss, Valid Acc, LR\n')
    log('Args: ' + str(args))
    log(f'Model full name: {model_fullname}')
    best_val_acc = 0.0
    best_epoch = 0
    epoch = 1
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train()
            val_loss, val_acc = evaluate()
            scheduler.step()
            log_and_save()
    except KeyboardInterrupt:
        log('*KeyboardInterrupt, early stop')
        args.epochs = epoch
        log_and_save()
    
    t = time.time() - start_time
    log(f'Training {epoch} epochs finished in {t/60:.2f} min, speed: {t/epoch:.1f} s/epoch')
