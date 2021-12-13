# -*- coding: utf-8 -*

import random
import time
import warnings
import sys
import argparse
import copy
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import os
import os.path as osp
import numpy as np

sys.path.append('./../')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.cls_domaindiscri import Class_Domaindiscri, rho_pred, binary_CE
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.transforms import ResizeImage
from tools.lr_scheduler import StepwiseLR
from tools.data_list import ImageList
from torch.autograd import Variable
from tools.epo_lp_dna import EPO_LP
from tools.TCM_loss import TCM


def get_current_time():
    time_stamp = time.time()  # 褰撳墠鏃堕棿鐨勬椂闂存埑
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d_%H-%M-%S', local_time)
    return str_time

def main(args: argparse.Namespace, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dset == "visda":
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    val_tranform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_source_dataset = ImageList(open(args.s_dset_path).readlines(), transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    train_target_dataset = ImageList(open(args.t_dset_path).readlines(), transform=train_transform)
    test_size = int(0.1 * len(train_target_dataset))
    train_size = len(train_target_dataset) - test_size
    target_dataset_train, target_dataset_valtidation = torch.utils.data.random_split(train_target_dataset,
                                                                                     [train_size, test_size])
    train_target_loader = DataLoader(target_dataset_train, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    validation_target_loader = DataLoader(target_dataset_valtidation, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers, drop_last=True)

    val_dataset = ImageList(open(args.t_dset_path).readlines(), transform=val_tranform)
    if args.dset == "visda":
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.dset == 'DomainNet':
        test_dataset = ImageList(open(args.t_dset_path).readlines(), transform=val_tranform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    validation_target_iter = ForeverDataIterator(validation_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    if args.dset == "office":
        num_classes = 31
    elif args.dset == "image-clef":
        num_classes = 12
    elif args.dset == "office-home":
        num_classes = 65
    elif args.dset == "visda":
        num_classes = 12

    classifier = ImageClassifier(backbone, num_classes).cuda()
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).cuda()
    cls_domain_discri = Class_Domaindiscri(embedding_dim=classifier.features_dim, num_domains=2,
                                           num_class=num_classes).cuda()
    epo_lp = EPO_LP(m=3)

    # define optimizer and lr scheduler
    optimizer_base = SGD(classifier.get_base_parameters_dict(),
                         args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler_base = StepwiseLR(optimizer_base, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    optimizer_specific = SGD(classifier.get_f_parameters_dict() + domain_discri.get_parameters(),
                             args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler_specific = StepwiseLR(optimizer_specific, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    optimizer_cd = SGD(cls_domain_discri.get_parameters(),
                       args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler_cd = StepwiseLR(optimizer_cd, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).cuda()

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, validation_target_iter, classifier, domain_adv, cls_domain_discri,
              optimizer_base, optimizer_specific, epo_lp,
              lr_scheduler_base, lr_scheduler_specific, optimizer_cd, lr_scheduler_cd, epoch, args)
        # evaluate on validation set
        if args.dset == "visda":
            acc1 = validate_visda(val_loader, classifier, epoch, config)
        else:
            acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)
        print("epoch= {:02d},  acc1={:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1))
        config["out_file"].write(
            "epoch = {:02d},  acc1 = {:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1) + '\n')
        config["out_file"].flush()

    print("best_acc1 = {:.3f}".format(best_acc1))
    config["out_file"].write("best_acc1 = {:.3f}".format(best_acc1) + '\n')
    config["out_file"].flush()

    # evaluate on test set
    classifier.load_state_dict(best_model)
    if args.dset == "visda":
        acc1 = validate_visda(test_loader, classifier, epoch, config)
    else:
        acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:.3f}".format(acc1))
    config["out_file"].write("test_acc1 = {:.3f}".format(acc1) + '\n')
    config["out_file"].flush()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          train_target_iter_meta: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, cls_domain_discri: Class_Domaindiscri,
          optimizer_base: SGD, optimizer_specific: SGD, epo_lp,
          lr_scheduler_base: StepwiseLR, lr_scheduler_specific: StepwiseLR, optimizer_cd: SGD,
          lr_scheduler_cd: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    total_losses = AverageMeter('total_loss', ':.3f')
    cls_losses = AverageMeter('cls_loss', ':.3f')
    trans_losses = AverageMeter('tran_loss', ':.3f')
    TCM_losses = AverageMeter('TCM_loss', ':.3f')
    cd_losses = AverageMeter('cd_loss', ':.3f')
    cls_accs = AverageMeter('Cls Acc', ':.3f')
    domain_accs = AverageMeter('Domain Acc', ':.3f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, total_losses, cls_losses, trans_losses, TCM_losses, cd_losses,
         cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()
    cls_domain_discri.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        n_linscalar_adjusts = 0
        descent = 0.
        grads = {}

        lr_scheduler_base.step()
        lr_scheduler_specific.step()
        lr_scheduler_cd.step()

        # get data
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)
        x_t_val, _ = next(train_target_iter_meta)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        x_t_val = x_t_val.cuda()
        labels_s = labels_s.cuda()

        ##epo learn
        optimizer_base.zero_grad()
        optimizer_specific.zero_grad()
        optimizer_cd.zero_grad()

        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        cd_s, cd_t = cls_domain_discri(f_s, f_t)  # logits
        rho_t = rho_pred(y_t, cd_t) # logits

        y_t_val, f_t_val = model(x_t_val)
        _, cd_t_val = cls_domain_discri(f_s, f_t_val)
        rho_t_val = rho_pred(y_t_val, cd_t_val)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        TCM_loss = -1 * TCM(rho_t)

        loss_val = -1*TCM(rho_t_val)

        ## task
        for t in range(3):
            optimizer_base.zero_grad()
            optimizer_specific.zero_grad()
            optimizer_cd.zero_grad()

            if t == 0:
                cls_loss.backward(retain_graph=True)
            elif t == 1:
                transfer_loss.backward(retain_graph=True)
            else:
                TCM_loss.backward()

            grads[t] = []
            for params in model.get_base_parameters():
                for param in params:
                    if param.grad is not None:
                        grads[t].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        G = torch.stack(grads_list)

        ## update pref
        optimizer_base.zero_grad()
        optimizer_specific.zero_grad()
        optimizer_cd.zero_grad()
        loss_val.backward()
        grads_val = []
        for params in model.get_base_parameters():
            for param in params:
                if param.grad is not None:
                    grads_val.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
        G_val = torch.cat(grads_val)

        try:
            # Calculate the alphas from the LP solver
            alpha = epo_lp.get_alpha(G=G.cpu().numpy(), G_val=G_val.cpu().numpy(), loss_val=loss_val)
            if epo_lp.last_move == "dom":
                descent += 1
        except Exception as e:
            print(e)
            alpha = None
        if alpha is None:  # A patch for the issue in cvxpy
            preference = np.array([args.pref1, args.pref2, args.pref3])
            alpha = preference
            n_linscalar_adjusts += 1

        alpha = 3 * torch.from_numpy(alpha).cuda()

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)  # logits
        f_s, f_t = f.chunk(2, dim=0)

        ##### class domain prediction #####
        cd_s, cd_t = cls_domain_discri(f_s, f_t)
        rho_t = rho_pred(y_t, cd_t)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        TCM_loss = -1 * TCM(rho_t)
        cd_loss = binary_CE(cd_s, cd_t, rho_t, labels_s)
        ####################################

        optimizer_base.zero_grad()
        optimizer_specific.zero_grad()
        optimizer_cd.zero_grad()

        total_loss = torch.stack([cls_loss, transfer_loss, TCM_loss], 0)
        total_loss = torch.sum(total_loss * alpha)
        total_loss.backward(retain_graph=True)

        optimizer_specific.zero_grad()
        optimizer_cd.zero_grad()
        total_loss1 = cls_loss + transfer_loss + cd_loss
        total_loss1.backward()

        optimizer_base.step()
        optimizer_specific.step()
        optimizer_cd.step()

        domain_acc = domain_adv.domain_discriminator_accuracy
        cls_acc = accuracy(y_s, labels_s)[0]
        total_losses.update(total_loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        cd_losses.update(cd_loss.item(), x_s.size(0))
        TCM_losses.update(TCM_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(alpha)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate_visda(val_loader, model, epoch, config):
    dict = {0: "plane", 1: "bcybl", 2: "bus", 3: "car", 4: "horse", 5: "knife", 6: "mcyle", 7: "person", 8: "plant", \
            9: "sktb", 10: "train", 11: "truck"}
    model.eval()
    with torch.no_grad():
        tick = 0
        subclasses_correct = np.zeros(12)
        subclasses_tick = np.zeros(12)
        for i, (imgs, labels) in enumerate(val_loader):
            tick += 1
            imgs = imgs.cuda()
            pred, _ = model(imgs)
            pred = nn.Softmax(dim=1)(pred)
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)
            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    subclasses_correct[pred[i]] += 1
        subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        print("Epoch [{:02d}]:".format(epoch))
        for i in range(12):
            log_str1 = '\t{}----------({:.3f})'.format(dict[i], subclasses_result[i] * 100.0)
            print(log_str1)
            config["out_file"].write(log_str1 + "\n")
        avg = subclasses_result.mean()
        avg = avg * 100.0
        log_avg = '\taverage:{:.3f}'.format(avg)
        print(log_avg)
        config["out_file"].write(log_avg + "\n")
        config["out_file"].flush()
    return avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('--arch', type=str, default='resnet101', choices=['resnet50', 'resnet101'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='7', help="device id to run")
    parser.add_argument('--dset', type=str, default='visda', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--train_path', type=str, default='data/visda2017/train_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--val_path', type=str, default='/data/visda2017/validation_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--output_dir', type=str, default='log/baseline/DANN/visda',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--iters-per-epoch', default=1000, type=int, help='Number of iterations per epoch')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')
    parser.add_argument('--pref1', type=float, default=1., help="pref1")
    parser.add_argument('--pref2', type=float, default=1., help="pref2")
    parser.add_argument('--pref3', type=float, default=1., help="pref3")
    args = parser.parse_args()

    config = {}
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task = args.s_dset_path.split('/')[-1].split('.')[0].split('_')[0] + "-" + \
           args.t_dset_path.split('/')[-1].split('.')[0].split('_')[0]
    config["out_file"] = open(osp.join(args.output_dir, get_current_time() + "_" + task + "_log.txt"), "w")

    config["out_file"].write("dann.py\n")
    import PIL

    config["out_file"].write("PIL version: {}\n".format(PIL.__version__))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        config["out_file"].write(str("{} = {}".format(arg, getattr(args, arg))) + "\n")
    config["out_file"].flush()
    main(args, config)

