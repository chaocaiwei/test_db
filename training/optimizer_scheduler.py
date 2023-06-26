import torch
from config import Configurable, State
import training.learning_rate


class OptimizerScheduler(Configurable):

    def __init__(self, optimizer, learning_rate, optimizer_args, cmd={}):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.optimizer_args = optimizer_args

        r_cl = learning_rate['class'].split('.')[-1]
        epochs = learning_rate['epochs']
        learning_rate.pop('class')
        self.learning_rate = getattr(training.learning_rate, r_cl)(**learning_rate)

        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.optimizer)(
                parameters, **self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
