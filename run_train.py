import argparse
import logging
import os
from pathlib import Path

import yaml
from torch.utils.tensorboard import SummaryWriter

from pyolov4.pyolov4.ddp import init_ddp
from pyolov4.pyolov4.evolve import evolve
from pyolov4.train import train
from pyolov4.utils.logger import setup_logging
from utils.general import (
    get_latest_run, check_git_status, check_file, increment_dir)
from utils.torch_utils import select_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov4.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='last', default=False,
                        help='resume from given file, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    arguments = parser.parse_args()

    setup_logging(Path("logging.conf"))
    logger = logging.getLogger(__name__)

    arguments.hyp = arguments.hyp or ('data/hyp.scratch.yaml')
    arguments.data, arguments.cfg, arguments.hyp = check_file(arguments.data), check_file(arguments.cfg), check_file(arguments.hyp)  # check files
    assert len(arguments.cfg) or len(arguments.weights), 'either --cfg or --weights must be specified'

    arguments.img_size.extend([arguments.img_size[-1]] * (2 - len(arguments.img_size)))  # extend to 2 sizes (train, test)
    device = select_device(arguments.device, batch_size=arguments.batch_size)
    arguments.total_batch_size = arguments.batch_size
    arguments.world_size = 1
    arguments.global_rank = -1

    # DDP mode
    if arguments.local_rank != -1:
        device, world_size, global_rank, batch_size = init_ddp(
            local_rank=arguments.local_rank,
            total_batch_size=arguments.batch_size
        )

    logger.info(arguments)
    with open(arguments.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    if not arguments.evolve:
        tb_writer = None
        if arguments.global_rank in [-1, 0]:
            logger.info(f"Start Tensorboard with `tensorboard --logdir {arguments.logdir}`, "
                        f"view at http://localhost:6006/")
            tb_writer = SummaryWriter(log_dir=increment_dir(Path(arguments.logdir) / 'exp', arguments.name))  # runs/exp

        train(hyp, arguments, device, tb_writer)

    else:
        # Evolve hyperparameters (optional)
        evolve(arguments)
