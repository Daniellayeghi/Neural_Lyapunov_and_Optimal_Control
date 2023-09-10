
class CustomLR:
    def __init__(self, optimizer,  halving_threshold, step_interval=10, gamma=0.5):
        self.steps_since_halving = 0
        self.should_halve = False
        self.halving_threshold = halving_threshold
        self.step_interval = step_interval
        self._not_halved = True
        self.optimizer = optimizer
        self._gamma = gamma

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def lr_lambda(self, epoch):
        if self.should_halve:
            self.steps_since_halving += 1
            if self.steps_since_halving % self.step_interval == 0 or self._not_halved:
                self.set_lr(self.get_lr() * self._gamma)
                self._not_halved = False

    def step(self, loss):
        if loss <= self.halving_threshold:
            self.should_halve = True
            self.lr_lambda(None)
