# -*- coding: utf-8 -*-
# flake8: noqa
import types
import warnings
import weakref

# import math                               # not required for LambdaMomentum
# from torch._six import inf                # not required for LambdaMomentum
from functools import wraps

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# from collections import Counter           # not required for LambdaMomentum
# from bisect import bisect_right           # not required for LambdaMomentum


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

SAVE_STATE_WARNING = "Please also save or load the state of the optimzer when saving or loading the scheduler."


class _MomentumScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base momentum
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_momentum", group["momentum"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_momentum" not in group:
                    raise KeyError(
                        "param 'initial_momentum' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_momenta = list(
            map(lambda group: group["initial_momentum"], optimizer.param_groups)
        )
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `momentum_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_momentum(self):
        """Return last computed momentum by current scheduler."""
        return self._last_momentum

    def get_momentum(self):
        # Compute momentum using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after momentum scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`momentum_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first momentum.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `momentum_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `momentum_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the momentum schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        class _enable_get_momentum_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_momentum_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_momentum_called_within_step = False

        with _enable_get_momentum_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_momentum()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_momentum"):
                    values = self._get_closed_form_momentum()
                else:
                    values = self.get_momentum()

        for param_group, momentum in zip(self.optimizer.param_groups, values):
            param_group["momentum"] = momentum

        self._last_momentum = [
            group["momentum"] for group in self.optimizer.param_groups
        ]


class LambdaMomentum(_MomentumScheduler):
    """Sets the momentum of each parameter group to the initial momentum
    times a given function. When last_epoch=-1, sets initial momentum as momentum.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        momentum_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaMomentum(optimizer, momentum_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, momentum_lambda, last_epoch=-1):
        self.optimizer = optimizer

        if not isinstance(momentum_lambda, list) and not isinstance(
            momentum_lambda, tuple
        ):
            self.momentum_lambdas = [momentum_lambda] * len(optimizer.param_groups)
        else:
            if len(momentum_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} momentum_lambdas, but got {}".format(
                        len(optimizer.param_groups), len(momentum_lambda)
                    )
                )
            self.momentum_lambdas = list(momentum_lambda)
        self.last_epoch = last_epoch
        super(LambdaMomentum, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The momentum lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """

        warnings.warn(SAVE_STATE_WARNING, UserWarning)
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "momentum_lambdas")
        }
        state_dict["momentum_lambdas"] = [None] * len(self.momentum_lambdas)

        for idx, fn in enumerate(self.momentum_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["momentum_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        warnings.warn(SAVE_STATE_WARNING, UserWarning)
        momentum_lambdas = state_dict.pop("momentum_lambdas")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["momentum_lambdas"] = momentum_lambdas

        for idx, fn in enumerate(momentum_lambdas):
            if fn is not None:
                self.momentum_lambdas[idx].__dict__.update(fn)

    def get_momentum(self):
        if not self._get_momentum_called_within_step:
            warnings.warn(
                "To get the last momentum computed by the scheduler, "
                "please use `get_last_momentum()`."
            )

        return [
            base_momentum * lmbda(self.last_epoch)
            for lmbda, base_momentum in zip(self.momentum_lambdas, self.base_momenta)
        ]
