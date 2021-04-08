import logging
import math
import os
import random
import test
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch.cuda
import torch.nn
import torch.tensor
import yaml
from torch import optim as optim, distributed as dist
from torch.cuda import amp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import tqdm

import pyolov4.pyolov4.evolve
from pyolov4.models.models import Darknet
from pyolov4.pyolov4.checkpoint import load_from_checkpoint
from pyolov4.utils.datasets import create_dataloader
from pyolov4.utils.general import check_img_size, labels_to_class_weights, plot_labels, \
    labels_to_image_weights, compute_loss, plot_images, fitness, strip_optimizer, plot_results
from pyolov4.utils.general import init_seeds
from pyolov4.utils.torch_utils import ModelEMA, torch_distributed_zero_first

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    weight_folder = log_dir / 'weights'   # weights directory
    weight_folder.mkdir(exist_ok=True)
    last = weight_folder / 'last.pt'
    best = weight_folder / 'best.pt'
    results_file = log_dir / 'results.txt'
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # TODO: Use DDP logging. Only the first process is allowed to log.

    # Save run settings
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds()

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    model = create_model(device, opt, rank, weights)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv2d.weight' in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0

    if opt.resume:
        best_fitness, start_epoch = load_from_checkpoint(best_fitness, epochs, model, opt, optimizer, results_file,
                                                         start_epoch)

    # Image sizes
    gs = 32 # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=(opt.local_rank))

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=opt.cache_images, rect=opt.rect, local_rank=rank,
                                            world_size=opt.world_size)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Testloader
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt, hyp=hyp, augment=False,
                                       cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)[0]

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Class frequency
    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

        # Check anchors
        #if not opt.noautoanchor:
        #    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dataset.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Autocast
            with amp.autocast(enabled=cuda):
                # Forward
                pred = model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                # if not torch.isfinite(loss):
                #     print('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=batch_size,
                                                 imgsz=imgsz_test,
                                                 save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not opt.nosave) or (final_epoch and not pyolov4.pyolov4.evolve.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if epoch >= (epochs-5):
                    torch.save(ckpt, last.replace('.pt','_{:03d}.pt'.format(epoch)))
                if (best_fitness == fi) and not final_epoch:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, weight_folder + 'last%s.pt' % n, weight_folder + 'best%s.pt' % n
        for f1, f2 in zip([weight_folder + 'last.pt', weight_folder + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
        # Finish
        if not pyolov4.pyolov4.evolve.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


def create_model(device, opt, rank, pretrained_weights: Optional[Path] = None):
    # Model
    with torch_distributed_zero_first(rank):
        model = Darknet(opt.cfg).to(device)  # create
        if pretrained_weights.is_file():
            weights = torch.load(pretrained_weights, map_location=device)  # load checkpoint
            state_dict = {k: v for k, v in weights.items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(state_dict, strict=False)
            logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), pretrained_weights))  # report
        else:
            logger.warning(f"File `{str(pretrained_weights)}` with weights don't found")
    return model