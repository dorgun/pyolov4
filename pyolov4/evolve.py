import os
import random
import time
from pathlib import Path

import numpy as np

from pyolov4.run_train import hyp, device
from pyolov4.train import train
from pyolov4.utils.general import fitness, print_mutation, plot_evolution


def evolve(arguments):
    # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'momentum': (0.1, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'giou': (1, 0.02, 0.2),  # GIoU loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (1, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (0, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (1, 0.0, 1.0),  # image flip left-right (probability)
            'mixup': (1, 0.0, 1.0)}  # image mixup (probability)
    assert arguments.local_rank == -1, 'DDP mode not implemented for --evolve'
    arguments.notest, arguments.nosave = True, True  # only test/save final epoch
    # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    yaml_file = Path('runs/evolve/hyp_evolved.yaml')  # save best result here
    if arguments.bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % arguments.bucket)  # download evolve.txt if exists
    for _ in range(100):  # generations to evolve
        if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
            # Select parent(s)
            parent = 'single'  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt('evolve.txt', ndmin=2)
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
            w = fitness(x) - fitness(x).min()  # weights
            if parent == 'single' or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == 'weighted':
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            mp, s = 0.9, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([x[0] for x in meta.values()])  # gains 0-1
            ng = len(meta)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                hyp[k] = float(x[i + 7] * v[i])  # mutate

        # Constrain to limits
        for k, v in meta.items():
            hyp[k] = max(hyp[k], v[1])  # lower limit
            hyp[k] = min(hyp[k], v[2])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        # Train mutation
        results = train(hyp.copy(), arguments, device)

        # Write mutation results
        print_mutation(hyp.copy(), results, yaml_file, arguments.bucket)
    # Plot results
    plot_evolution(yaml_file)
    print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
          'hyperparameters: $ python run_train.py --hyp %s' % (yaml_file, yaml_file))