import os
import logging
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageNet
import wandb

from classifier import Classifier
from image_list import ImageList
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    ProgressMeter,
)


def get_source_optimizer(model, args):
    if args.distributed:
        model = model.module
    backbone_params, extra_params = model.get_params()
    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": extra_params,
                    "lr": args.optim.lr * 10,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]

    return optimizer


def train_source_domain(args):
    logging.info(f"Start source training on {args.data.src_domain}...")

    model = Classifier(args.model_src).to("cuda")
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    logging.info(f"1 - Created source model")

    # transforms
    train_transform = get_augmentation("plain")
    val_transform = get_augmentation("test")

    # datasets
    if args.data.dataset == "imagenet-1k":
        train_dataset = ImageNet(args.data.image_root, transform=train_transform)
        val_dataset = ImageNet(
            args.data.image_root, split="val", transform=val_transform
        )
    else:
        label_file = os.path.join(
            args.data.image_root, f"{args.data.src_domain}_list.txt"
        )
        train_dataset = ImageList(
            args.data.image_root, label_file, transform=train_transform
        )
        val_dataset = ImageList(
            args.data.image_root, label_file, transform=val_transform
        )
        assert len(train_dataset) == len(val_dataset)

        # split the dataset with indices
        indices = np.random.permutation(len(train_dataset))
        num_train = int(len(train_dataset) * args.data.train_ratio)
        train_dataset = Subset(train_dataset, indices[:num_train])
        val_dataset = Subset(val_dataset, indices[num_train:])
    logging.info(
        f"Loaded {len(train_dataset)} samples for training "
        + f"and {len(val_dataset)} samples for validation",
    )

    # data loaders
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.data.workers,
    )
    val_sampler = DistributedSampler(val_dataset) if args.distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.data.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=args.data.workers,
    )
    logging.info(f"2 - Created data loaders")

    optimizer = get_source_optimizer(model, args)
    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info(f"3 - Created optimizer")

    logging.info(f"Start training...")
    best_acc = 0.0
    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, optimizer, epoch, args)

        # evaluate
        accuracy = evaluate(val_loader, model, domain=args.data.src_domain, args=args)
        if accuracy > best_acc and is_master(args):
            best_acc = accuracy
            filename = f"best_{args.data.src_domain}_{args.seed}.pth.tar"
            save_path = os.path.join(args.log_dir, filename)
            save_checkpoint(model, optimizer, epoch, save_path=save_path)

    # evaluate on target before any adaptation
    for t, tgt_domain in enumerate(args.data.target_domains):
        if tgt_domain == args.data.src_domain:
            continue
        label_file = os.path.join(args.data.image_root, f"{tgt_domain}_list.txt")
        tgt_dataset = ImageList(args.data.image_root, label_file, val_transform)
        sampler = DistributedSampler(tgt_dataset) if args.distributed else None
        tgt_loader = DataLoader(
            tgt_dataset,
            batch_size=args.data.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=args.data.workers,
        )

        logging.info(f"Evaluate {args.data.src_domain} model on {tgt_domain}")
        evaluate(
            tgt_loader,
            model,
            domain=f"{args.data.src_domain}-{tgt_domain}",
            args=args,
            wandb_commit=(t == len(args.data.target_domains) - 1),
        )


def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    loss = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, loss, top1], prefix="Epoch: [{}]".format(epoch),
    )

    # make sure to switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        images = data[0].cuda(args.gpu, non_blocking=True)
        labels = data[1].cuda(args.gpu, non_blocking=True)

        # per-step scheduler
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)

        logits = model(images)

        loss_ce = smoothed_cross_entropy(
            logits,
            labels,
            num_classes=args.model_src.num_classes,
            epsilon=args.learn.epsilon,
        )

        # train acc measure (on one GPU only)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().detach() * 100.0
        loss.update(loss_ce.item(), images.size(0))
        top1.update(acc.item(), images.size(0))

        if use_wandb(args):
            wandb.log({"Loss": loss_ce.item()}, commit=(i != len(train_loader)))

        # perform one gradient step
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)


def evaluate(val_loader, model, domain, args, wandb_commit=True):
    model.eval()

    logging.info(f"Evaluating...")
    gt_labels, all_preds = [], []
    with torch.no_grad():
        iterator = tqdm(val_loader) if is_master(args) else val_loader
        for data in iterator:
            images = data[0].cuda(args.gpu, non_blocking=True)
            labels = data[1]

            logits = model(images)
            preds = logits.argmax(dim=1).cpu()

            gt_labels.append(labels)
            all_preds.append(preds)

    gt_labels = torch.cat(gt_labels)
    all_preds = torch.cat(all_preds)

    if args.distributed:
        gt_labels = concat_all_gather(gt_labels.cuda())
        all_preds = concat_all_gather(all_preds.cuda())

        ranks = len(val_loader.dataset) % dist.get_world_size()
        gt_labels = remove_wrap_arounds(gt_labels, ranks).cpu()
        all_preds = remove_wrap_arounds(all_preds, ranks).cpu()

    accuracy = (all_preds == gt_labels).float().mean() * 100.0
    wandb_dict = {f"{domain} Acc": accuracy}

    logging.info(f"Accuracy: {accuracy:.2f}")
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.numpy(), y_pred=all_preds.numpy()
        )
        wandb_dict[f"{domain} Avg"] = acc_per_class.mean()
        wandb_dict[f"{domain} Per-class"] = acc_per_class

    if use_wandb(args):
        wandb.log(wandb_dict, commit=wandb_commit)

    return accuracy


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss
