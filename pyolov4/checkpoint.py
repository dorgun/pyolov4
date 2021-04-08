from pathlib import Path

from pyolov4.train import logger
from pyolov4.utils.general import get_latest_run


def load_from_checkpoint(best_fitness, epochs, model, opt, optimizer, results_file, start_epoch):
    # Resume
    if opt.resume:
        ckpt_file = Path(
            get_latest_run(opt.logdir) if opt.resume == 'last' else opt.resume)  # resume from most recent run
        if ckpt_file.is_file():
            logger.info(f'Resuming training from file `{str(ckpt_file)}`')
            ckpt = torch.load(ckpt_file)

            # Optimizer
            if optimizer in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            else:
                logger.warning("Optimizer state not found in checkpoint file.")

            # best loss
            if best_fitness in ckpt:
                best_fitness = ckpt['best_fitness']
            else:
                logger.warning("Best loss result not found in checkpoint file.")

            # Results
            if 'training_results' in ckpt:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt
            else:
                logger.warning("Training result not found in checkpoint file.")

            # Epochs
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
            else:
                logger.warning("Last epoch not found in checkpoint file.")

            if 'model' in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                logger.critical("Model state not found in checkpoint file.")
                exit()

            if epochs < start_epoch:
                logger.info(f"Model from {str(ckpt_file)} has been trained for {start_epoch - 1} epochs. "
                            f"Fine-tuning for {epochs - start_epoch - 1} additional epochs.")
            else:
                logger.info("Reach the maximum number of epoch.")
            del ckpt
    return best_fitness, start_epoch