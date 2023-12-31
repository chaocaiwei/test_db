import os

import torch
from tqdm import tqdm

from experiment import Experiment


class Trainer:
    def __init__(self, experiment: Experiment, validate=False):
        self.init_device()

        self.experiment = experiment
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver
        self.args = experiment.args

        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.current_lr = 0

        self.total = 0

        if validate:
            self.validation = self.experiment.validation
        else:
            self.validation = None




    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        return self.experiment.structure.model

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.experiment.train.scheduler.learning_rate.get_learning_rate(
            epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

    def train(self):
        self.logger.report_time('Start')
        self.logger.args(self.args)
        model = self.init_model()
        train_data_loader = self.experiment.train.data_loader
        if self.validation:
            validation_loaders = self.experiment.validation.data_loaders

        self.steps = 0
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta

        # Init start epoch and iter
        optimizer = self.experiment.train.scheduler.create_optimizer(
            model.parameters())

        self.logger.report_time('Init')

        model.train()
        while True:
            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)

            for batch in train_data_loader:

                self.logger.report_time("Data loading")

                if self.validation and\
                        self.steps % self.validation.interval == 0 and\
                        self.steps > self.validation.exempt:
                    self.validate(validation_loaders, model, epoch, self.steps)
                self.logger.report_time('Validating ')
                if self.logger.verbose and torch.cuda.is_available():
                    torch.cuda.synchronize()

                self.train_step(model, optimizer, batch,
                                epoch=epoch, step=self.steps)
                self.update_learning_rate(optimizer, epoch, self.steps)

                if self.logger.verbose and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.logger.report_time('Forwarding ')

                self.model_saver.maybe_save_model(
                    model, epoch, self.steps, self.logger)

                self.steps += 1
                self.logger.report_eta(self.steps, self.total, epoch)

            epoch += 1
            if epoch > self.experiment.train.epochs:
                self.model_saver.save_checkpoint(model, 'final')
                if self.validation:
                    self.validate(validation_loaders, model, epoch, self.steps)
                self.logger.info('Training done')
                break
            iter_delta = 0

    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        optimizer.zero_grad()

        pred = model.forward(batch)
        results = self.experiment.structure.loss(pred, batch)
        if len(results) == 2:
            l, metrics = results
        elif len(results) == 3:
            l, pred, metrics = results

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l.mean()
        loss.backward()
        optimizer.step()

        if step % self.experiment.logger.log_interval == 0:
            if isinstance(l, dict):
                line = '\t'.join(line)
                log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(step, epoch, line, self.current_lr)
                self.logger.info(log_info)
            else:
                self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                    step, epoch, loss.item(), self.current_lr))
            self.logger.add_scalar('loss', loss, step)
            self.logger.add_scalar('learning_rate', self.current_lr, step)
            for name, metric in metrics.items():
                self.logger.add_scalar(name, metric.mean(), step)
                self.logger.info('%s: %6f' % (name, metric.mean()))

            self.logger.report_time('Logging')

    def validate(self, loader, model, epoch, step):
        all_matircs = {}
        model.eval()
        name = loader.dataset.dataset_name
        if self.validation.visualize:
            metrics, vis_images = self.validate_step(
                    loader, model, True)
            self.logger.images(
                    os.path.join('vis', name), vis_images, step)
        else:
            metrics, vis_images = self.validate_step(loader, model, False)
        for _key, metric in metrics.items():
            key = name + '/' + _key
            if key in all_matircs:
                all_matircs[key].update(metric.val, metric.count)
            else:
                all_matircs[key] = metric

        for key, metric in all_matircs.items():
            self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
        self.logger.metrics(epoch, step, all_matircs)
        model.train()
        return all_matircs

    def validate_step(self, data_loader, model, visualize=False):
        raw_metrics = []
        vis_images = dict()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            pred = model.forward(batch)
            output = self.structure.representer.represent(batch, pred)
            result = self.structure.measurer.validate_measure(
                batch, output)
            if len(result) == 1:
                raw_metric = result
            elif len(result) == 2:
                raw_metric, interested = result
            raw_metrics.append(raw_metric)

            if visualize and self.structure.visualizer:
                if interested is not  None:
                    vis_image = self.structure.visualizer.visualize(
                    batch, output, interested)
                    vis_images.update(vis_image)
        metrics = self.structure.measurer.gather_measure(
            raw_metrics, self.logger)
        return metrics, vis_images

    def to_np(self, x):
        return x.cpu().data.numpy()
