import os.path as osp
from collections import OrderedDict
import math
from .clip import tokenize
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast


class OriClip(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.clip = clip_model
        self.text_input = torch.cat([tokenize(f"a photo of a {c}") for c in classnames]).to(cfg.device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.clip.encode_image(image.type(self.dtype))
        text_features = self.clip.encode_text(self.text_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features


# f1w1
class ZeroShotOriClip(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def task_adapt(self, classnames, t):
        # now_classnames = classnames[: (t+1) * self.cfg.dataset.n_classes_per_task]
        text_input = torch.cat([tokenize(f"a photo of a {c}") for c in classnames]).to(self.cfg.device)
        return text_input

    def forward_features(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def forward_mlp(self, image_features, text_input):
        image_features = self.mlp_visual(image_features)
        text_features = self.clip.encode_text(text_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, text_input

    def forward(self, image, text_input):
        image_features = self.clip.visual(image.type(self.dtype))
        text_features = self.clip.encode_text(text_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features


# f2w1
class MLPCustomClip(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip = clip_model
        embed_dim = self.clip.visual.output_dim
        self.mlp_visual = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(embed_dim, embed_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(embed_dim // 16, embed_dim))
        ]))
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward_features(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def forward_mlp(self, image_features, text_input):
        image_features = self.mlp_visual(image_features)
        text_features = self.clip.encode_text(text_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, text_input

    def forward(self, image, text_input):
        image_features = self.clip.visual(image.type(self.dtype))
        image_features = self.mlp_visual(image_features)
        text_features = self.clip.encode_text(text_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features


# (f1 + f2) * w1
class ResMLPCustomClip(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip = clip_model
        embed_dim = self.clip.visual.output_dim
        self.mlp_visual = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(embed_dim, embed_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(embed_dim // 16, embed_dim))
        ]))
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ratio = 1

    def forward_features(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def forward_mlp(self, image_features, text_input):
        x = self.mlp_visual(image_features)
        image_features = self.ratio * x + self.ratio * image_features
        text_features = self.clip.encode_text(text_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, text_input

    def forward(self, image, text_input):
        image_features = self.clip.visual(image.type(self.dtype))
        x = self.mlp_visual(image_features)
        image_features = self.ratio * x + self.ratio * image_features
        text_features = self.clip.encode_text(text_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features
