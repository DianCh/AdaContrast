#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import logging
import os
import random

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import group
import torch.optim
import torch.multiprocessing as mp
import wandb

from source import train_source_domain
from target import train_target_domain as train_target_adacontrast
from utils import configure_logger, NUM_CLASSES, use_wandb


@hydra.main(config_path="configs", config_name="root")
def main(args):
    # enable adding attributes at runtime
    OmegaConf.set_struct(args, False)
    args.job_name = HydraConfig.get().job.name

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # seed each process
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set process specific info
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.gpu)

        # adjust data settings according to multi-processing
        args.data.batch_size = int(args.data.batch_size / args.ngpus_per_node)
        args.data.workers = int(
            (args.data.workers + args.ngpus_per_node - 1) / args.ngpus_per_node
        )

    work_dir = os.getcwd()
    os.makedirs(work_dir, exist_ok=True)
    args.log_dir = work_dir
    configure_logger(args.rank)
    logging.info(
        f"Dataset: {args.data.dataset},"
        + f" Source domains: {args.data.source_domains},"
        + f" Target domains: {args.data.target_domains},"
        + f" Pipeline: {'source' if args.train_source else 'target'}"
    )

    # iterate over each domain
    args.data.image_root = os.path.join(args.data.data_root, args.data.dataset)
    args.model_src.num_classes = NUM_CLASSES[args.data.dataset]
    if args.train_source:
        for src_domain in args.data.source_domains:
            args.data.src_domain = src_domain

            if use_wandb(args):
                wandb.init(
                    project=args.project if args.project else args.data.dataset,
                    group=args.memo,
                    job_type=src_domain,
                    name=f"seed_{args.seed}",
                    config=dict(args),
                )
            # main loop
            train_source_domain(args)
            if use_wandb(args):
                wandb.finish()
    else:
        for src_domain in args.data.source_domains:
            args.data.src_domain = src_domain
            for tgt_domain in args.data.target_domains:
                if src_domain == tgt_domain:
                    continue
                args.data.tgt_domain = tgt_domain

                if use_wandb(args):
                    wandb.init(
                        project=args.project if args.project else args.data.dataset,
                        group=args.memo,
                        job_type=f"{src_domain}-{tgt_domain}-{args.sub_memo}",
                        name=f"seed_{args.seed}",
                        config=dict(args),
                    )
                # main loop
                if args.target_algorithm == "ours":
                    train_target_adacontrast(args)
                if use_wandb(args):
                    wandb.finish()


if __name__ == "__main__":
    main()
