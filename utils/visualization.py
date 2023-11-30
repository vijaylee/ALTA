import torch
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Union
import matplotlib.colors as mcolors



def plot_tsne(args, method, labels, embeds, idx, t, save_path=None, **kwargs: Dict[str, Any]):
    """t-SNE visualize
    Args:
        labels (Tensor): labels of test and train
        embeds (Tensor): embeds of test and train
        name ([str], optional): same as <name> in roc_auc. Defaults to None.
        save_path ([str], optional): same as <name> in roc_auc. Defaults to None.
        kwargs (Dict[str, Any]): hyper parameters of t-SNE which will change final result
            n_iter (int): > 250, default = 1000
            learning_rate (float): (10-1000), default = 100
            perplexity (float): (5-50), default = 28
            early_exaggeration (float): change it when not converging, default = 12
            angle (float): (0.2-0.8), default = 0.3
            init (str): "random" or "pca", default = "pca"
    """
    tsne = TSNE(
        n_components=2,
        verbose=1,
        n_iter=kwargs.get("n_iter", 1000),
        learning_rate=kwargs.get("learning_rate", 100),
        perplexity=kwargs.get("perplexity", 28),
        early_exaggeration=kwargs.get("early_exaggeration", 12),
        angle=kwargs.get("angle", 0.3),
        init=kwargs.get("init", "pca"),
    )
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)

    css4 = list(mcolors.CSS4_COLORS.keys())
    # color_ind = [10, 14]
    color_ind = [19,2,9,10,11,13,14,16,17,1,20,21,25,28,30,31,32,37,38,40,47,51,
                55,60,65,82,85,88,106,110,115,118,120,125,131,135,139,142,146,147]
    css4 = [css4[v] for v in color_ind]

    legends = [f'{i}' for i in range(100)]
    (_, ax) = plt.subplots(1)
    plt.title(f't-SNE: {args.dataset.il_setting}_{args.model.visual}_{method}_{args.dataset.name}_bf{args.model.buffer_size}')
    for label in torch.unique(labels):
        res = tsne_results[torch.where(labels==label)]
        ax.plot(*res.T, marker="o", linestyle="", ms=5, color=css4[label])
        # show legend
        # ax.plot(*res.T, marker="o", linestyle="", ms=5, label=legends[label], color=css4[label])
        # ax.legend(loc="best")
    plt.xticks([])
    plt.yticks([])

    out_path = save_path if save_path else './tsne_results'
    os.makedirs(out_path, exist_ok=True)
    if t < args.dataset.n_tasks:
        image_path = os.path.join(out_path,
                                  f'{args.dataset.il_setting}_{idx}th-{args.model.visual}_{method}_{args.dataset.name}-task{t}_bf{args.model.buffer_size}'+'_tsne.pdf')
    else:
        image_path = os.path.join(out_path,
                                  f'{args.dataset.il_setting}_{idx}th-{args.model.visual}_{method}_{args.dataset.name}_bf{args.model.buffer_size}' + '_tsne.pdf')
    plt.savefig(image_path)
    plt.close()
    return