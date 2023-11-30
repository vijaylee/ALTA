import torch

def mask_classes(args, outputs: torch.Tensor, idx: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if 'fewshot' in args.dataset.name and idx == 0:
        N_CLASSES_PER_TASK = args.dataset.base_set
    else:
        N_CLASSES_PER_TASK = args.dataset.n_classes_per_task

    outputs[:, 0:idx * N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (idx + 1) * N_CLASSES_PER_TASK:
               args.dataset.n_tasks * N_CLASSES_PER_TASK] = -float('inf')


def mask_class_names(args, class_names, idx: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if 'fewshot' in args.dataset.name and idx == 0:
        N_CLASSES_PER_TASK = args.dataset.base_set
    else:
        N_CLASSES_PER_TASK = args.dataset.n_classes_per_task

    t_class_names = class_names[idx * N_CLASSES_PER_TASK: (idx + 1) * N_CLASSES_PER_TASK]
    return t_class_names