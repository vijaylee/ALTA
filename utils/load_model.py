import torch
from collections import OrderedDict
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import tolist_if_not
import sys
sys.path.append("..")
from models import clip, clip_model


def load_clip_to_cpu(args):
    backbone_name = args.visual
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip_model.build_model(state_dict or model.state_dict())
    return model


class TrainerBase:
    """Base class for iterative trainer."""
    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )
        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )
        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )
        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def update_lr(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()
