import torch
from config import Configurable, State
import training.learning_rate


class OptimizerScheduler(Configurable):

    def __init__(self, optimizer, learning_rate, optimizer_args, cmd={}):

        r_cl = learning_rate['class'].split('.')[-1]
        learning_rate.pop('class')
        learning_rate['lr'] = optimizer_args['lr']

        if 'lr' in cmd:
            optimizer_args['lr'] = cmd['lr']
            learning_rate['lr'] = cmd['lr']

        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.learning_rate = getattr(training.learning_rate, r_cl)(**learning_rate)



    def create_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.optimizer)(
                parameters, **self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
