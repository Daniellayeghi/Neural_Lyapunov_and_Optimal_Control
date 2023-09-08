from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, halving_threshold, step_interval=10, last_epoch=-1):
        self.halving_threshold = halving_threshold
        self.step_interval = step_interval
        self.steps_since_halving = 0
        self.should_halve = False
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.should_halve:
            self.steps_since_halving += 1
            if self.steps_since_halving % self.step_interval == 0:
                return [group['lr'] / 2.0 for group in self.optimizer.param_groups]

        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, loss):
        if loss <= self.halving_threshold:
            self.should_halve = True

        super(CustomLRScheduler, self).step()
