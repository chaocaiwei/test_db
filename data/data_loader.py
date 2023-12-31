import math
import bisect

import imgaug
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, ConcatDataset, BatchSampler
from config import Configurable, State
from data.image_datasets import ImageDataset
import data.processes.data_process
from data.processes.augment_data import *


def default_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    imgaug.seed(worker_id)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=256, num_workers=10, is_training=True, shuffle=True, collect_fn=None, drop_last=True, cmd={}):
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.collect_fn = collect_fn
        self.drop_last = drop_last

        if 'batch_size' in cmd:
            self.batch_size = cmd['batch_size']
        if 'num_workers' in cmd:
            self.num_workers = cmd['num_workers']

        if self.collect_fn is None:
            self.collect_fn = torch.utils.data.dataloader.default_collate
        if self.shuffle is None:
            self.shuffle = self.is_training

        self.dataset = self.load_datasets(dataset, cmd)
        torch.utils.data.DataLoader.__init__(
                self, self.dataset,
                batch_size=self.batch_size, num_workers=self.num_workers,
                drop_last=self.drop_last, shuffle=self.shuffle,
                pin_memory=True, collate_fn=self.collect_fn,
                worker_init_fn=default_worker_init_fn,
                generator=torch.Generator(device='cpu'))
        self.collect_fn = str(self.collect_fn)

    def load_datasets(self, datasets, cmd):
        if 'processes' in datasets:
            processed = self.load_processes(datasets['processes'])
            dataset_name = datasets['dataset_name']
            data_dir = datasets['data_dir']
            if 'data_dir' in cmd:
                data_dir = cmd['data_dir']
            datasets = ImageDataset(processes=processed, is_training=self.is_training, dataset_name=dataset_name, data_dir=data_dir, cmd=cmd)
        return datasets

    def load_processes(self, processes):
        procs = []
        for p in processes:
            cls = p['class'].split('.')[-1]
            if cls == 'AugmentDetectionData':
                proc = AugmentDetectionData(p)
                procs.append(proc)
            else:
                proc = getattr(data.processes.data_process, cls)(p)
                procs.append(proc)
        return procs


class SuccessiveRandomSampler(Sampler):
    '''Random Sampler that yields sorted data in successive ranges.
    Args:
        dataset: Dataset used for sampling.
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InfiniteOrderedSampler(Sampler):
    def __init__(self, data_source, limit_size):
        self.data_source = data_source
        self.limit_size = limit_size

    def __iter__(self):
        n = len(self.data_source)

        def wrapper():
            cnt = 0
            while cnt < self.limit_size:
                if cnt % n == 0:
                    idx = torch.randperm(n).tolist()
                yield idx[cnt % n]
                cnt += 1
        return wrapper()

    def __len__(self):
        return self.limit_size



class RandomSampleSampler(Sampler):
    def __init__(self, data_source, weights=None, size=2 ** 31):
        self.data_source = data_source
        if weights is None:
            self.probabilities = np.full(len(data_source), 1 / len(data_source))
        else:
            self.probabilities = np.array(weights) / np.sum(weights)
        self.cum_prob = np.cumsum(self.probabilities)
        self.size = size

    def __iter__(self):
        def wrapper():
            for i in range(self.size):
                yield bisect.bisect(self.cum_prob, torch.rand(1)[0], hi=len(self.data_source) - 1)
        return wrapper()

    def __len__(self):
        return self.size
