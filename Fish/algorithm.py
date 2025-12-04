import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from numbers import Number
import operator
from Fish.model import WholeFish
from torch.autograd import Variable


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


class Fish(nn.Module):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, x_dim, h_dim, y_dim, args):
        super(Fish, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        self.args = args

        self.network = WholeFish(x_dim, h_dim, y_dim)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=args.LR,
            weight_decay=args.weight_decay
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = WholeFish(
            self.x_dim, self.h_dim, self.y_dim, weights=self.network.state_dict()
        ).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.args.LR,
            weight_decay=self.args.weight_decay
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def pre_update(self, loaders, device):
        loss_t = 0
        for loader in loaders:
            for labels, ppg, scr, scl, eda in loader:
                labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)
                loss = F.cross_entropy(self.network(ppg, eda), labels)
                self.optimizer.zero_grad()
                loss_t += loss.item()
                loss.backward()
                self.optimizer.step()
        return {'loss': loss_t/len(loaders)}

    def update(self, loaders, device):
        self.create_clone(device)
        loss_t = 0
        for loader in loaders:
            for labels, ppg, scr, scl, eda in loader:
                labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)
                loss = F.cross_entropy(self.network_inner(ppg, eda), labels)
                self.optimizer_inner.zero_grad()
                loss.backward()
                loss_t += loss.item()
                self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.args.meta_lr
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss_t/len(loaders)}

    def predict(self, ppg, eda):
        return self.network(ppg, eda)
