import os
import torch


class Checkpoint:

    def __init__(self, start_epoch=0, start_iter=0, resume=None, cmd={}):
        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.resume = resume

        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            self.logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        logger.info("Resuming from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter
