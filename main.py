import os
import torch
import torch.nn.functional as F
from torch import nn
from datasets import get_dataloaders, get_classnames
from arguments import get_args
from tqdm import tqdm
from models import convert_weights, MLPCustomClip, ResMLPCustomClip, SPCustomCoOp, CPCustomCoOp, CPOriPCustomCoOp, CPMLPCustomCoOp,\
    CPResMLPCustomCoOp, OriCLIP2CPMLPCustomCoOp, OriTextMLP2CPOriImgCustomCoOp
from models.clip import tokenize
from utils import build_optimizer, build_lr_scheduler, FeaBuffer, mask_classes, load_clip_to_cpu, TrainerBase
from dassl.engine import TRAINER_REGISTRY



@TRAINER_REGISTRY.register()
class MyCLIP(TrainerBase):
    method2model = {'mlp': MLPCustomClip, 'resmlp': ResMLPCustomClip, 'sprompt': SPCustomCoOp, 'cprompt': CPCustomCoOp,
                    'cprompt_oriprompt': CPOriPCustomCoOp, 'cprompt_mlp': CPMLPCustomCoOp, 'cprompt_resmlp': CPResMLPCustomCoOp,
                    'oriclip2cprompt_mlp': OriCLIP2CPMLPCustomCoOp, 'oritext_mlp2cprompt_oriimg': OriTextMLP2CPOriImgCustomCoOp}

    def build_model(self, args):
        classes_name = get_classnames(args)
        myclip_model = load_clip_to_cpu(args)
        model = self.method2model[args.method](args, classes_name, myclip_model)

        if args.model.pretrained:
            print("Turning off gradients in both the image and the text encoder")
            if args.method in ['mlp', 'resmlp']:
                name_to_update = "mlp_visual"
                optim_module = model.mlp_visual
            else:
                name_to_update = "prompt_learner"
                optim_module = model.prompt_learner
            for name, param in model.named_parameters():
                if name_to_update not in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        convert_weights(model)
        model.to(args.device)

        optimizer = build_optimizer(optim_module, args.OPTIM)
        scheduler = build_lr_scheduler(optimizer, args.OPTIM)
        self.register_model(name_to_update, optim_module, optimizer, scheduler)
        return model, optimizer, optim_module


def main(args):
    myclip = MyCLIP()
    model, optimizer, optim_module = myclip.build_model(args)
    if args.method in ['sprompt', 'cprompt', 'cprompt_oriprompt']:
        args.model.use_replay = False
    buffer = FeaBuffer(args, args.buffer_size, args.device)
    buffer.to(args.device)

    dataloaders_test = []
    classes_names = get_classnames(args)
    text_input = torch.cat([tokenize(f"a photo of a {c}") for c in classes_names]).to(args.device)
    for t in range(args.dataset.n_tasks):
        print('---' * 10, f'Task:{t}', '---' * 10)
        train_dataloader, dataloaders_test, data_train_nums = get_dataloaders(args, t, dataloaders_test)
        model.eval()
        optim_module.train()
        if args.method not in ['mlp', 'resmlp']:
            if t > 0:
                text_input = prompts[:t * args.dataset.n_classes_per_task]
            else:
                text_input = torch.Tensor([])
        for epoch in range(args.OPTIM.MAX_EPOCH):
            with tqdm(train_dataloader, unit="batch") as tepoch:
                total_nums, epoch_correct, epoch_loss = 0, 0.0, 0.0
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    optimizer.zero_grad()
                    if args.method in ['sprompt', 'cprompt', 'cprompt_oriprompt']:
                        logits, prompts = model(inputs, text_input)
                    else:
                        image_feas = model.forward_features(inputs)
                        logits, prompts = model.forward_mlp(image_feas, text_input)
                    loss = F.cross_entropy(logits, labels)
                    if not buffer.is_empty():
                        buf_labels, buf_logits, buf_image_features = buffer.get_data(args.train.batch_size)
                        psat_logits, _ = model.forward_mlp(buf_image_features, text_input)
                        loss += args.train.alpha * F.cross_entropy(psat_logits, buf_labels) + args.train.beta * nn.MSELoss()(psat_logits, buf_logits)

                    epoch_loss += loss.item()
                    predict = torch.argmax(logits.data, 1)
                    total_nums += inputs.shape[0]
                    epoch_correct += torch.sum(predict == labels).item()
                    loss.backward()
                    optimizer.step()
                    if args.model.use_replay:
                        buffer.add_data(labels=labels, logits=logits.data, image_features=image_feas.data)
                    accuracy = epoch_correct / total_nums
                    tepoch.set_postfix(loss=epoch_loss, acc='{:.3f}'.format(accuracy))


                if args.eval.test_epochs > 0 and (epoch + 1) % args.eval.test_epochs == 0:
                    optim_module.eval()
                    eval_total_acc = 0.0
                    for idx, dataloader_test in enumerate(dataloaders_test):
                        correct, total = 0.0, 0.0
                        with torch.no_grad():
                            for x, y in dataloader_test:
                                x, y = x.to(args.device), y.to(args.device)
                                logits, _ = model(x, text_input)
                                if args.il_setting == 'task-il':
                                    mask_classes(args, logits, idx)
                                pred = torch.argmax(logits.data, 1)
                                correct += torch.sum(pred == y).item()
                                total += y.shape[0]
                            acc = correct / total
                            eval_total_acc += acc
                            print('---model---', 'task:', idx, 'acc:', acc, '**' * 11)
                    eval_mean_acc = eval_total_acc / (idx + 1)
                    print('---model---', 'mean_acc:', eval_mean_acc, '**' * 11)
                    optim_module.train()



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args = get_args()
    main(args)



