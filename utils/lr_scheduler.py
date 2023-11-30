"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    max_epoch = optim_cfg.MAX_EPOCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(max_epoch)
    )
    if optim_cfg.WARMUP_EPOCH > 0:
        if not optim_cfg.WARMUP_RECOUNT:
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )
    return scheduler

def build_base_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    max_epoch = optim_cfg.BASE_MAX_EPOCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(max_epoch)
    )
    if optim_cfg.WARMUP_EPOCH > 0:
        if not optim_cfg.WARMUP_RECOUNT:
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )
    return scheduler

