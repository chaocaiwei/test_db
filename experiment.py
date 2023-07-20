from structure.model import SegDetectorModel
import torch
from decoders.seg_detector_loss import L1BalanceCELoss
from torch import nn
from structure.represent import SegDetectorRepresenter
from structure.measurer import QuadMeasurer
from structure.visualizer import SegDetectorVisualizer
from data.data_loader import DataLoader
from training.checkpoint import Checkpoint
from training.log import Logger
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from training.learning_rate import *

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Structure:

    def __init__(self, model, representer=None, measurer=None, visualizer=None, cmd={}):

        arg = model['model_args']
        backbone = arg['backbone']
        decoder = arg['decoder']
        decoder_args = arg['decoder_args']
        self.model = SegDetectorModel(backbone=backbone, decoder=decoder, decoder_args=decoder_args, device=try_gpu())

        self.loss = nn.DataParallel(L1BalanceCELoss())

        if representer is not None:
            max_candidates = representer['max_candidates']
            self.representer = SegDetectorRepresenter(max_candidates=max_candidates, cmd=cmd)
        if measurer is not None:
            self.measurer = QuadMeasurer()
        if visualizer is not None:
            if 'eager_show' in visualizer:
                eager_show = visualizer['eager_show']
                self.visualizer = SegDetectorVisualizer(eager_show=eager_show)
            else:
                self.visualizer = SegDetectorVisualizer()


class Train:

    def __init__(self, epochs, data_loader, checkpoint, model_saver, scheduler, cmd={}):
        self.epochs = epochs
        dataset = data_loader['dataset']
        batch_size = data_loader['batch_size']
        num_workers = data_loader['num_workers']
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size, num_workers, is_training=True, cmd=cmd)

        start_epoch = checkpoint['start_epoch']
        start_iter = checkpoint['start_iter']
        resume = checkpoint['resume']
        self.checkpoint = Checkpoint(start_epoch, start_iter, resume, cmd)

        dir_path = model_saver['dir_path']
        save_interval = model_saver['save_interval']
        signal_path = model_saver['signal_path']
        if 'out_path' in cmd and cmd['out_path'] is not None:
            dir_path = cmd['out_path'] + dir_path
            signal_path = cmd['out_path'] + signal_path
        self.model_saver = ModelSaver(dir_path, save_interval, signal_path)

        optimizer = scheduler['optimizer']
        optimizer_args = scheduler['optimizer_args']
        learning_rate = scheduler['learning_rate']
        learning_rate['epochs'] = epochs
        self.scheduler = OptimizerScheduler(optimizer, learning_rate, optimizer_args, cmd=cmd)


class Validation:

    def __init__(self, data_loader, visualize=False, interval=4500, exempt=None, cmd={}):
        dataset = data_loader['dataset']
        batch_size = data_loader['batch_size']
        num_workers = data_loader['num_workers']
        self.batch_size = batch_size
        self.data_loaders = DataLoader(dataset, batch_size, num_workers, is_training=False, cmd=cmd)

        self.visualize = visualize
        self.interval = interval
        self.exempt = exempt


class Experiment:

    def __init__(self, args):
        self.args = args
        cmd = args['cmd']
        exp = cmd['exp'].split('.')[0]

        structure = args['structure']
        model = structure['model']
        representer = structure['representer']
        measurer = structure['measurer']
        visualizer = structure['visualizer']
        self.structure = Structure(model, representer, measurer, visualizer, cmd=cmd)

        train = args['train']
        epochs = train['epochs']
        data_loader = train['data_loader']
        checkpoint = train['checkpoint']
        model_saver = train['model_saver']
        scheduler = train['scheduler']
        if 'epochs' in cmd:
            epochs = cmd['epochs']
        self.train = Train(epochs, data_loader, checkpoint, model_saver, scheduler, cmd=cmd)

        log = args['logger']
        verbose = log['verbose']
        level = log['level']
        log_interval = log['log_interval']
        log_dir = 'workspace'
        if 'out_path' in cmd and cmd['out_path'] is not None:
            log_dir = cmd['out_path'] + log_dir
        self.logger = Logger(name=exp, verbose=verbose, log_dir=log_dir, level=level, log_interval=log_interval)

        validation = args['validation']
        data_loader = validation['data_loader']
        visualize = validation['visualize']
        interval = validation['interval']
        exempt = validation['exempt']
        self.validation = Validation(data_loader, visualize, interval, exempt, cmd)

