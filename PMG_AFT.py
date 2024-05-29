from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
import datetime
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, SubsetRandomSampler
# from torchvision.datasets import CIFAR100, CIFAR10, Caltech101, STL10, OxfordIIITPet, DTD

from torchvision.datasets import *

import torchvision.transforms as transforms
import torchvision

import clip
from models import prompters
from models.prompters import TokenPrompter, NullPrompter
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint  # , CLASS_SPLIT_CIFAR100
from utils import cosine_lr, convert_models_to_fp32, refine_classname

import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import logging
from autoattack import AutoAttack
from PIL import Image


def parse_option():
    parser = argparse.ArgumentParser('Pre-trained-Model-Guided-Fine-Tuning for CLIP')

    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--test_freq', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--train_eps', type=float, default=2)
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=1)
    parser.add_argument('--test_numsteps', type=int, default=10)
    parser.add_argument('--test_stepsize', type=int, default=1)
    parser.add_argument('--earlystop', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--imagenet_root', type=str, default=None)
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch',
                        choices=['null_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=0,
                        help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--image_size', type=int, default=224)

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--evaluate', default=False, action="store_true", )
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--Noattack', action='store_true')
    parser.add_argument('--CW', action='store_true')

    parser.add_argument('--train_class_count', type=int, default=90)
    parser.add_argument('--last_num_ft', type=int, default=-1)

    parser.add_argument('--noimginprop', action='store_true')
    parser.add_argument('--autoattack', action='store_true')
    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}_addp_{}'. \
        format(args.name, args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial,
               args.add_prompt_size)

    return args


best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

fname = 'FT_reg'
if not os.path.exists(fname):
    os.makedirs(fname)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(fname, str(datetime.datetime.now()) + 'output.log')),
        logging.StreamHandler()
    ])

ImageNet_MEAN = (0.485, 0.456, 0.406)
ImageNet_STD = (0.229, 0.224, 0.225)

mu_img = torch.tensor(ImageNet_MEAN).view(3, 1, 1).cuda()
std_img = torch.tensor(ImageNet_STD).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu_img) / std_img


def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)

    return X


upper_limit, lower_limit = 1, 0


def get_indices(dataset, num_samples_per_class):
    indices = []
    class_counts = {}  # 记录每个类别的样本数
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_counts:
            class_counts[label] = 0
        if class_counts[label] < num_samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
    return indices


# Custom transform to ensure all images are RGB
def to_rgb(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def main():
    global best_acc1, device

    args = parse_option()
    args.train_eps = args.train_eps / 255.
    args.test_eps = args.test_eps / 255.
    args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = args.test_stepsize / 255.
    logger.info('!!!!!!This is a new test!!!!!')
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    imagenet_root = '/data/wangsibo/ImageNet'
    tinyimagenet_root = '/data/wangsibo/tinyImageNet/tiny-imagenet-200'
    imgnet_full = imagenet_root

    if args.imagenet_root is not None:
        imagenet_root = args.imagenet_root

    add_prompt_len = 0

    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
    model_ori, preprocess_ori = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
    model_text, model_image = None, None

    convert_models_to_fp32(model_ori)
    model_ori = torch.nn.DataParallel(model_ori)
    model_ori.eval()

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()

    prompter = NullPrompter()
    add_prompter = TokenPrompter(add_prompt_len)

    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    if args.last_num_ft == -1:
        optimizer = torch.optim.SGD(model.module.visual.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(list(model.module.visual.parameters())[-args.last_num_ft:],
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction="sum").to(device)
    args.start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    logger.info(f'template: {template}')

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    preprocess224_a = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    preprocess224 = transforms.Compose([
        transforms.Lambda(lambda image: to_rgb(image)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    preprocess224_interpolate = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    preprocess112_interpolate = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    if args.dataset == 'cifar100':
        train_dataset = CIFAR100(args.root, transform=preprocess224,
                                 download=True, train=True)

        val_dataset = CIFAR100(args.root, transform=preprocess,
                               download=True, train=False)
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(args.root, transform=preprocess,
                                download=True, train=True)

        val_dataset = CIFAR10(args.root, transform=preprocess,
                              download=True, train=False)

    elif args.dataset == 'ImageNet':
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(imagenet_root, 'train'),
            transform=preprocess224
        )

    elif args.dataset == 'tinyImageNet':
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(tinyimagenet_root, 'train'),
            transform=preprocess224_a)

    val_dataset_list = []
    if args.evaluate:
        val_dataset_name = ['cifar10', 'cifar100', 'STL10', 'SUN397', 'Food101',
                            'oxfordpet', 'flowers102', 'dtd', 'EuroSAT', 'fgvc_aircraft',
                            'tinyImageNet', 'ImageNet', 'Caltech101', 'Caltech256', 'StanfordCars', 'PCAM']

    else:
        val_dataset_name = ['cifar10', 'cifar100', 'STL10', 'SUN397', 'Food101',
                            'oxfordpet', 'flowers102', 'dtd', 'EuroSAT', 'fgvc_aircraft',
                            'tinyImageNet', 'ImageNet', 'Caltech101', 'Caltech256', 'StanfordCars', 'PCAM']
    for each in val_dataset_name:
        if each == 'cifar10':
            val_dataset_list.append(CIFAR10(args.root, transform=preprocess,
                                            download=True, train=False))
        elif each == 'cifar100':
            val_dataset_list.append(CIFAR100(args.root, transform=preprocess,
                                             download=True, train=False))

        elif each == 'Caltech101':
            val_dataset_list.append(Caltech101(args.root, target_type='category', transform=preprocess224,
                                               download=False))
        elif each == 'PCAM':
            val_dataset_list.append(PCAM(args.root, split='test', transform=preprocess224,
                                         download=False))
        elif each == 'STL10':
            val_dataset_list.append(STL10(args.root, split='test',
                                          transform=preprocess, download=True))
        elif each == 'SUN397':
            val_dataset_list.append(SUN397(args.root,
                                           transform=preprocess224, download=True))
        elif each == 'StanfordCars':
            val_dataset_list.append(StanfordCars(args.root, split='test',
                                                 transform=preprocess224, download=False))
        elif each == 'Food101':
            val_dataset_list.append(Food101(args.root, split='test',
                                            transform=preprocess224, download=True))
        elif each == 'oxfordpet':
            val_dataset_list.append(OxfordIIITPet(args.root, split='test',
                                                  transform=preprocess224, download=True))
        elif each == 'EuroSAT':
            val_dataset_list.append(EuroSAT(args.root,
                                            transform=preprocess224, download=True))

        elif each == 'Caltech256':
            val_dataset_list.append(Caltech256(args.root, transform=preprocess224,
                                               download=False))
        elif each == 'flowers102':
            val_dataset_list.append(Flowers102(args.root, split='test',
                                               transform=preprocess224, download=True))
        elif each == 'dtd':
            val_dataset_list.append(DTD(args.root, split='test',
                                        transform=preprocess224, download=True))

        elif each == 'fgvc_aircraft':
            val_dataset_list.append(FGVCAircraft(args.root, split='test',
                                                 transform=preprocess224, download=True))

        elif each == 'ImageNet':
            val_dataset_list.append(torchvision.datasets.ImageFolder(
                os.path.join(imgnet_full, 'val'),
                transform=preprocess224))

        elif each == 'tinyImageNet':
            val_dataset_list.append(torchvision.datasets.ImageFolder(
                os.path.join(tinyimagenet_root, 'val'),
                transform=preprocess224))

    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True, sampler=train_sampler)

    val_loader_list = [DataLoader(each,
                                  batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=val_sampler) for each in
                       val_dataset_list]

    class_names = train_dataset.classes

    if args.dataset == 'ImageNet' or args.dataset == 'tinyImageNet':
        from utils import load_imagenet_folder2name
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names

    class_names = refine_classname(class_names)
    texts_train = [template.format(label) for label in class_names]

    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
        else:
            class_names = each.classes
            if val_dataset_name[cnt] == 'ImageNet' or val_dataset_name[cnt] == 'tinyImageNet':
                from utils import load_imagenet_folder2name
                folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                new_class_names = []
                for class_name in class_names:
                    new_class_names.append(folder2name[class_name])
                class_names = new_class_names

            class_names = refine_classname(class_names)
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)

    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    if args.evaluate:
        acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                             prompter, add_prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.start_epoch, args.epochs):

        start = time.time()
        # train for one epoch
        train(train_loader, texts_train, model, model_ori, model_text, model_image, prompter, add_prompter, optimizer,
              scheduler, criterion, criterion_kl, scaler, epoch, args)
        end = time.time()
        logger.info(f"Time for one epoch: {end - start}")
        l2_norm_obj = sum(p.norm(2) for p in model.module.visual.parameters())
        l2_norm_ori = sum(p.norm(2) for p in model_ori.module.visual.parameters())
        ratio = abs(l2_norm_ori - l2_norm_obj) / float(l2_norm_ori)
        abs_l2 = abs(l2_norm_ori - l2_norm_obj)
        logger.info(f"ratio, l2: {ratio, abs_l2}")
        # evaluate on validation set
        if epoch % args.test_freq == 0:
            acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                                 prompter, add_prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1_mean > best_acc1
        best_acc1 = max(acc1_mean, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'vision_encoder_state_dict': model.module.visual.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            logger.info(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.earlystop:
                logger.info("The training halted by early stopping criterion.")
                break


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


from utils import one_hot_embedding


def attack_CW(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
              attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _, _, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_CW_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                       attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        _images = clip_img_preprocessing(X + delta)
        # output, _ = model(_images, text_tokens)

        output, _, _, _ = multiGPU_CLIP(model_image, model_text, model, _images, text_tokens, None)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _, _, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_pgd_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):

        _images = clip_img_preprocessing(X + delta)
        output, _, _, _ = multiGPU_CLIP(model_image, model_text, model, _images, text_tokens, None)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
    img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
    scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
    logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
    logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
    return logits_per_image, logits_per_text, img_embed, scale_text_embed


def train(train_loader, texts, model, model_ori, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, criterion, criterion_kl, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.module.visual.train()

    num_batches_per_epoch = len(train_loader)

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        BATCH_SIZE = images.size(0)
        # logger.info('bs', BATCH_SIZE)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)
        # logger.info(images.min(), images.max())

        # with automatic mixed precision
        with autocast():
            if not args.Noattack:
                delta = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, images,
                                   target, text_tokens, alpha, attack_iters, 'l_inf', epsilon=args.train_eps)
                # logger.info('delta', delta.min(), delta.max())

                tmp = clip_img_preprocessing(images + delta)
            else:
                tmp = clip_img_preprocessing(images)

            tem_clean = clip_img_preprocessing(images)
            prompted_images = prompter(tmp)
            prompted_clean_images = prompter(tem_clean)
            prompt_token = None

            # for multiple GPU
            output, _, img_embed, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens,
                                                    prompt_token)
            output_ori, _, img_embed_ori, _ = multiGPU_CLIP(model_image, model_text, model_ori, prompted_images,
                                                            text_tokens,
                                                            prompt_token)
            output_clean, _, img_embed_clean, _ = multiGPU_CLIP(model_image, model_text, model, prompted_clean_images,
                                                                text_tokens,
                                                                prompt_token)

            loss_advori = criterion_kl(F.log_softmax(output, dim=1), F.softmax(output_ori, dim=1))
            loss_advclean = criterion_kl(F.log_softmax(output, dim=1), F.softmax(output_clean, dim=1))
            loss = criterion(output, target) + loss_advclean + loss_advori
            # print(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger.info(progress)

        if i % args.print_freq == 0:
            progress.display(i)
            if args.debug:
                break

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'vision_encoder_state_dict': model.module.visual.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    logger.info(
        ' * Time :{} Data :{}  loss:{} acc@1:{}'
            .format(batch_time, data_time, losses.avg, top1.avg))

    return losses.avg, top1.avg


# def validate(val_loader, texts, model, prompter, add_prompter, criterion, args):
def validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
             prompter, add_prompter, criterion, args):
    dataset_num = len(val_loader_list)
    acc_all = []

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')
        top1_adv_prompt = AverageMeter('Adv Prompt Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_prompt, top1_adv_org, top1_adv_prompt],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()

        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            if 'cifar' not in val_dataset_name:
                if i % 20 != 0 and not args.evaluate:
                    continue

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            # logger.info(images.size())

            with autocast():

                # clean images, with prompt and without prompt
                # compute output
                with torch.no_grad():
                    # prompt_token = add_prompter()
                    prompt_token = None
                    # output_prompt, _ = model(prompter(clip_img_preprocessing(images)), text_tokens, prompt_token)
                    output_prompt, _, _, _ = multiGPU_CLIP(model_image, model_text, model,
                                                           prompter(clip_img_preprocessing(images)), text_tokens,
                                                           prompt_token)

                    loss = criterion(output_prompt, target)

                    # measure accuracy and record loss
                    acc1 = accuracy(output_prompt, target, topk=(1,))
                    losses.update(loss.item(), images.size(0))
                    top1_prompt.update(acc1[0].item(), images.size(0))

                    top1_org.update(acc1[0].item(), images.size(0))

                torch.cuda.empty_cache()

                # generate adv example
                if args.CW:
                    delta_prompt = attack_CW(prompter, model, model_text, model_image, add_prompter, criterion,
                                             images, target, text_tokens,
                                             test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                elif args.autoattack:
                    def model_fn(x):
                        output_a, _, _, _ = multiGPU_CLIP(model_image, model_text, model,
                                                          prompter(clip_img_preprocessing(x)),
                                                          text_tokens,
                                                          prompt_token)
                        return output_a.to(torch.float32)

                    adversary = AutoAttack(model_fn, norm='Linf', eps=args.test_eps, version='standard')
                    adv_samples = adversary.run_standard_evaluation(images, target, bs=100)
                    delta_prompt = adv_samples - images
                    delta_prompt = clamp(delta_prompt, lower_limit - images, upper_limit - images)
                else:
                    delta_prompt = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion,
                                              images, target, text_tokens,
                                              test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)

                # compute output
                torch.cuda.empty_cache()
                with torch.no_grad():
                    prompt_token = add_prompter()
                    # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
                    output_prompt_adv, _, _, _ = multiGPU_CLIP(model_image, model_text, model,
                                                               prompter(clip_img_preprocessing(images + delta_prompt)),
                                                               text_tokens, prompt_token)

                    loss = criterion(output_prompt_adv, target)

                # bl attack
                torch.cuda.empty_cache()

                # measure accuracy and record loss
                acc1 = accuracy(output_prompt_adv, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_adv_prompt.update(acc1[0].item(), images.size(0))

                acc1 = accuracy(output_prompt_adv, target, topk=(1,))
                top1_adv_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        torch.cuda.empty_cache()
        logger.info('A new test!')
        logger.info(
            dataset_name + ' * Adv Prompt Acc@1 {top1_adv_prompt.avg:.3f} Adv Original Acc@1 {top1_adv_org.avg:.3f} '
                           '*  Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
            .format(top1_adv_prompt=top1_adv_prompt, top1_adv_org=top1_adv_org,
                    top1_prompt=top1_prompt, top1_org=top1_org))
        acc_all.append(top1_adv_prompt.avg)

    return np.mean(acc_all)


if __name__ == '__main__':
    main()
