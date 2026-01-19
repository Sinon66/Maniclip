import argparse
import os
import pickle
import numpy as np
import random
import warnings
import time
from collections import OrderedDict
from typing import Optional

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn import LayerNorm
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import math
import clip
import shutil

from external.stylegan2.model import Generator
from external.stylegan2.calc_inception import load_patched_inception_v3

from utils.id_loss import IDLoss
from utils.utils import adjust_learning_rate, AverageMeter, ProgressMeter, calc_fid, int_item, parse_mask
from utils.average_lab_color_loss import AvgLabLoss
from utils.data_processing import produce_labels
from utils.model_irse import IRSE


HISTOGRAM_WARNING_TAGS = set()


def safe_add_histogram(writer, tag, values, step, fallback_values=None):
    if writer is None:
        return
    try:
        writer.add_histogram(tag, values, step)
    except Exception as exc:
        if fallback_values is not None:
            try:
                writer.add_histogram(tag, fallback_values, step)
                return
            except Exception as fallback_exc:
                exc = fallback_exc
        if tag not in HISTOGRAM_WARNING_TAGS:
            warnings.warn(f"Skipping histogram {tag} due to error: {exc}")
            HISTOGRAM_WARNING_TAGS.add(tag)


parser = argparse.ArgumentParser(description='ManiCLIP Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=31, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='training mini-batch size')
parser.add_argument('--test_batch', default=50, type=int, metavar='N',
                    help='test batchsize (default: 50)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N',
                    help='print frequency (default: 30)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=10, type=int, help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--task_name', default='model', type=str, help='task name')
parser.add_argument('--loss_clip_weight', type=float, default=1.0,
                    help='The clip loss for optimization. (default: 1.0)')
parser.add_argument('--loss_w_norm_weight', type=float, default=0.01,
                    help='The L2 loss on the latent codes w for optimization. (default: 0.01)')
parser.add_argument('--loss_minmaxentropy_weight', type=float, default=0,
                    help='The entropy loss on the latent codes w for optimization. (default: 0.)')
parser.add_argument('--loss_id_weight', type=float, default=0,
                    help='The ID loss for optimization. (default: 0.3)')
parser.add_argument('--loss_face_bg_weight', type=float, default=0.,
                    help='The face background loss for optimization. (default: 1.0)')
parser.add_argument('--loss_face_norm_weight', type=float, default=0.,
                    help='The loss between the input and output face area colous. (default: 0.)')
parser.add_argument("--truncation", type=float, default=0.7, help="truncation ratio")
parser.add_argument("--truncation_mean", type=int, default=4096,
                    help="number of vectors to calculate mean for the truncation")
parser.add_argument("--ckpt", type=str, default="pretrained/ffhq_256.pt",
                    help="path to the model checkpoint")
parser.add_argument("--size", type=int, default=256, help="output image size of the generator")
parser.add_argument('--decouple', action='store_true',
                    help='Use decoupling training scheme')
parser.add_argument("--part_sample_num", default=3, type=int,
                    help="the number of attributes sampled for each text segment")
parser.add_argument("--protect_attr_idx", default=20, type=int,
                    help="protected attribute index for subgroup statistics")
parser.add_argument('--use_state_mod', action='store_true',
                    help='enable state modulation gate on offsets (default: False)')
parser.add_argument('--log_g_only', action='store_true',
                    help='compute and log g without affecting losses (default: False)')
parser.add_argument('--adv_weight', type=float, default=0.1,
                    help='weight for adversarial GRL loss (default: 0.1)')
parser.add_argument('--use_adv', action='store_true',
                    help='enable adversarial GRL loss (default: False)')
parser.add_argument('--adv_apply_to', type=str, choices=['base', 'both'], default=None,
                    help='apply adversarial loss to base branch or both (default: base, or both with counterfactual)')
parser.add_argument('--grl_lambda', type=float, default=1.0,
                    help='gradient reversal lambda (default: 1.0)')
parser.add_argument('--adv_warmup_steps', type=int, default=500,
                    help='warmup steps to linearly ramp adv_weight (default: 500)')
parser.add_argument('--use_counterfactual', action='store_true',
                    help='enable counterfactual keep-edit training (default: False)')
parser.add_argument('--counterfactual_every', type=int, default=1,
                    help='run keep branch every N steps when use_counterfactual is enabled (default: 1)')
parser.add_argument('--cf_microbatch', type=int, default=0,
                    help='counterfactual microbatch size (0 disables microbatching)')
parser.add_argument('--w_keep', type=float, default=0.5,
                    help='weight for keep loss in counterfactual training (default: 0.5)')
parser.add_argument('--keep_text_cache_size', type=int, default=4096,
                    help='max size for keep_text_cache LRU (0 disables cache)')
parser.add_argument('--train_num', type=int, default=None,
                    help='(train_min) override train split length when provided')
parser.add_argument('--val_num', type=int, default=None,
                    help='(train_min) override val split length when provided')


class CLIPLoss(nn.Module):
    def __init__(self, clip_model):
        super(CLIPLoss, self).__init__()
        self.model = clip_model

    def forward(self, image, text):
        image = torchvision.transforms.functional.resize(image, 224)
        distance = 1 - self.model(image, text)[0] / 100
        return distance


def init_parsing_model(args):
    """
    Only needed if you use parse_mask() losses (bg / face_norm).
    """
    from external.parsing import BiSeNet
    args.parse_model = BiSeNet(n_classes=19)
    args.parse_model.load_state_dict(torch.load('external/parsing/models/bisenet.pth'))
    args.parse_model = args.parse_model.cuda()
    args.parse_model.eval()


def safe_write_text_lines(fp, sampled_text, max_lines=9):
    """
    sampled_text may be a string or a list/tuple of strings.
    This helper writes up to max_lines safely without IndexError.
    """
    if isinstance(sampled_text, str):
        sampled_text = [sampled_text]
    if sampled_text is None:
        return
    # in case it's something else (e.g., numpy array)
    try:
        n = len(sampled_text)
    except Exception:
        sampled_text = [str(sampled_text)]
        n = 1

    for k in range(min(max_lines, n)):
        fp.write(f"{k}: {sampled_text[k]}\n")


def main():
    args = parser.parse_args()
    if args.counterfactual_every < 1:
        raise ValueError("counterfactual_every must be >= 1")
    if args.cf_microbatch < 0:
        raise ValueError("cf_microbatch must be >= 0")
    if args.use_adv and not args.use_state_mod:
        raise ValueError(
            "use_adv requires use_state_mod because adversarial loss is defined on gated offset (offset')"
        )
    if args.adv_apply_to is None:
        args.adv_apply_to = 'both' if args.use_counterfactual else 'base'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting from checkpoints.'
        )

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    args.save_folder = os.path.join('models', args.task_name)
    os.makedirs(args.save_folder, exist_ok=True)

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    args.clip_model, _ = clip.load("ViT-B/32", device="cuda")
    clip_loss = CLIPLoss(args.clip_model)

    # ID loss is created even if weight=0; harmless.
    args.id_loss = IDLoss().cuda().eval()

    face_model = IRSE()
    face_model = nn.DataParallel(face_model).cuda()
    checkpoint = torch.load('pretrained/attribute_model.pth.tar')
    face_model.load_state_dict(checkpoint['state_dict'])
    face_model.eval()
    args.face_model = face_model

    # create model
    model = TransModel(nhead=8, num_decoder_layers=6)
    model.clip_model = model.clip_model.float()
    print(model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    discriminator = None
    if args.use_adv:
        discriminator = SubgroupDiscriminator(input_dim=14 * 512).cuda(args.gpu)

    optimizer_params = list(model.parameters())
    if discriminator is not None:
        optimizer_params += list(discriminator.parameters())

    optimizer = torch.optim.Adam(
        optimizer_params,
        args.lr,
        betas=(0.5, 0.999)
    )

    # optionally resume
    if args.resume:
        print(f"=> loading checkpoint '{args.resume}'")
        load_path = os.path.join(args.resume)
        if args.gpu is None:
            checkpoint = torch.load(load_path)
        else:
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(load_path, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if discriminator is not None and 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    cudnn.benchmark = True

    # Train loader
    if args.decouple:
        dataset_train = PartTextDataset(
            split='train',
            sample_num=args.part_sample_num,
            train_num=args.train_num,
            val_num=args.val_num,
        )
    else:
        dataset_train = TextDataset(split='train', train_num=args.train_num, val_num=args.val_num)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    # Eval loader: always uses TextDataset in this codebase
    dataset_eval = TextDataset(split='eval', train_num=args.train_num, val_num=args.val_num)
    eval_loader = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    generator = Generator(args.size, 512, 8).cuda()
    generator.load_state_dict(torch.load(args.ckpt)['g_ema'], strict=False)
    generator.eval()

    with open('pretrained/inception_ffhq.pkl', 'rb') as f:
        embeds = pickle.load(f)
        args.real_mean = embeds['mean']
        args.real_cov = embeds['cov']

    args.inception = nn.DataParallel(load_patched_inception_v3()).cuda()
    args.inception.eval()

    args.ce_noreduced = nn.CrossEntropyLoss(reduce=False).cuda()
    args.ce_criterion = nn.CrossEntropyLoss().cuda()
    args.average_color_loss = AvgLabLoss().cuda().eval()

    fid_best = float("inf")

    # IMPORTANT: parsing model only needed if those losses are enabled
    if args.loss_face_bg_weight or args.loss_face_norm_weight:
        init_parsing_model(args)

    if args.truncation < 1:
        with torch.no_grad():
            args.mean_latent = generator.mean_latent(args.truncation_mean)
    else:
        args.mean_latent = None

    if args.evaluate:
        with torch.no_grad():
            fid = validate(eval_loader, model, None, generator, clip_loss, 0, args)
        return

    log_path = os.path.join('logs', args.task_name)
    writter = SummaryWriter(log_path)

    iteration_num = args.start_epoch * len(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, discriminator, writter, generator, clip_loss, optimizer, epoch, args,
              iteration_num=iteration_num)
        iteration_num += len(train_loader)

        with torch.no_grad():
            fid = validate(eval_loader, model, writter, generator, clip_loss, epoch, args)

        if fid < fid_best:
            fid_best = fid
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if discriminator is not None:
                checkpoint_state['discriminator_state_dict'] = discriminator.state_dict()
            save_checkpoint(checkpoint_state, is_best=True, save_folder=args.save_folder)
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if discriminator is not None:
                checkpoint_state['discriminator_state_dict'] = discriminator.state_dict()
            save_checkpoint(checkpoint_state, is_best=False, save_folder=args.save_folder)


def save_checkpoint(state, is_best, save_folder, filename='latest.pth.tar'):
    filename = os.path.join(save_folder, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_folder, 'model_best.pth.tar'))


def train(train_loader, model, discriminator, writter, generator, clip_loss, optimizer, epoch, args, iteration_num=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    clip_losses = AverageMeter('clip_loss', ':.4e')
    bg_losses = AverageMeter('bg_loss', ':.4e')
    w_norm_losses = AverageMeter('w_norm_loss', ':.4e')
    id_losses = AverageMeter('id_loss', ':.4e')
    face_norm_losses = AverageMeter('face_norm_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    adv_losses = AverageMeter('adv_loss', ':.4e')
    keep_losses = AverageMeter('keep_loss', ':.4e')
    protect_flip_rates = AverageMeter('protect_flip_rate', ':.4e')
    all_losses = AverageMeter('all_losses', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, all_losses, clip_losses, w_norm_losses, id_losses, face_norm_losses, entropy_losses,
         adv_losses, keep_losses],
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()
    if discriminator is not None:
        discriminator.train()
    end = time.time()
    g_total = 0
    g_ones = 0
    keep_text_cache = OrderedDict()
    cache_enabled = args.keep_text_cache_size > 0
    print(f"keep_text_cache enabled: {cache_enabled} / size: {args.keep_text_cache_size}")

    log_g = args.log_g_only or args.use_state_mod or args.use_adv or args.use_counterfactual
    for i, (clip_text, sampled_text, labels, exist_mask, length) in enumerate(train_loader):
        data_time.update(time.time() - end)

        global_step = iteration_num + i
        writter.add_scalar('Train/flags_use_state_mod', int(args.use_state_mod), global_step)
        writter.add_scalar('Train/flags_use_adv', int(args.use_adv), global_step)
        writter.add_scalar('Train/flags_use_counterfactual', int(args.use_counterfactual), global_step)
        cf_enabled_this_step = args.use_counterfactual and (global_step % args.counterfactual_every == 0)
        writter.add_scalar('Train/cf_enabled_this_step', int(cf_enabled_this_step), global_step)
        writter.add_scalar('Train/cf_microbatch', args.cf_microbatch, global_step)
        if args.use_counterfactual and not cf_enabled_this_step and i % args.print_freq == 0:
            print(f"[Step {global_step}] Counterfactual keep skipped (counterfactual_every={args.counterfactual_every}).")

        def _slice_sampled_text(sampled, start, end):
            if isinstance(sampled, (list, tuple)):
                return sampled[start:end]
            return sampled

        batch_size = clip_text.size(0)
        microbatch_size = args.cf_microbatch if args.cf_microbatch > 0 else batch_size
        microbatch_size = min(microbatch_size, batch_size)

        optimizer.zero_grad()
        for mb_start in range(0, batch_size, microbatch_size):
            mb_end = min(batch_size, mb_start + microbatch_size)
            mb_count = mb_end - mb_start
            clip_text_mb = clip_text[mb_start:mb_end]
            labels_mb = labels[mb_start:mb_end]
            exist_mask_mb = exist_mask[mb_start:mb_end]
            sampled_text_mb = _slice_sampled_text(sampled_text, mb_start, mb_end)

            if args.gpu is not None:
                clip_text_mb = clip_text_mb.cuda(args.gpu, non_blocking=True)
                labels_mb = labels_mb.cuda(args.gpu, non_blocking=True)
                exist_mask_mb = exist_mask_mb.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                code = torch.randn(mb_count, 512).cuda()
                styles = generator.style(code)
                input_im, _ = generator([styles], input_is_latent=True, randomize_noise=False,
                                        truncation=args.truncation, truncation_latent=args.mean_latent)

                g = None
                if log_g:
                    in_attr = args.face_model(torchvision.transforms.functional.resize(input_im, 256))
                    in_preds = torch.stack(in_attr).transpose(0, 1).argmax(-1)
                    g = in_preds[:, args.protect_attr_idx]

            clip_text_base = clip_text_mb
            clip_text_keep = None
            if args.use_counterfactual and cf_enabled_this_step:
                if isinstance(sampled_text_mb, (list, tuple)):
                    sampled_text_base = [str(text) for text in sampled_text_mb]
                else:
                    sampled_text_base = [str(sampled_text_mb)]
                keep_tokens = []
                for text in sampled_text_base:
                    if cache_enabled:
                        cached = keep_text_cache.get(text)
                        if cached is None:
                            cached = clip.tokenize([f"{text}, keep gender unchanged"], truncate=True)
                            keep_text_cache[text] = cached
                            keep_text_cache.move_to_end(text)
                            if len(keep_text_cache) > args.keep_text_cache_size:
                                keep_text_cache.popitem(last=False)
                        else:
                            keep_text_cache.move_to_end(text)
                        keep_tokens.append(cached)
                    else:
                        keep_tokens.append(clip.tokenize([f"{text}, keep gender unchanged"], truncate=True))
                clip_text_keep = torch.cat(keep_tokens, dim=0)
                if args.gpu is not None:
                    clip_text_keep = clip_text_keep.cuda(args.gpu, non_blocking=True)

            offset_base = model(styles, clip_text_base, g if args.use_state_mod else None)
            delta_base = offset_base
            delta_base_flat = delta_base.reshape(delta_base.size(0), -1)
            if args.use_state_mod:
                s = model.last_s
                s_computed = s is not None
                writter.add_scalar('Train/s_computed_this_step', int(s_computed), global_step)
                if s_computed:
                    writter.add_scalar('Train/s_mean', s.mean().item(), global_step)
                    writter.add_scalar('Train/s_std', s.std().item(), global_step)
            else:
                writter.add_scalar('Train/s_computed_this_step', 0, global_step)
            new_styles = styles.unsqueeze(1).repeat(1, 14, 1) + offset_base

            gen_im_base, _ = generator([new_styles], input_is_latent=True, randomize_noise=False,
                                       truncation=args.truncation, truncation_latent=args.mean_latent)
            if args.use_counterfactual and cf_enabled_this_step:
                offset_keep = model(styles, clip_text_keep, g if args.use_state_mod else None)
                delta_keep = offset_keep
                delta_keep_flat = delta_keep.reshape(delta_keep.size(0), -1)
                new_styles_keep = styles.unsqueeze(1).repeat(1, 14, 1) + offset_keep
                gen_im_keep, _ = generator([new_styles_keep], input_is_latent=True, randomize_noise=False,
                                           truncation=args.truncation, truncation_latent=args.mean_latent)

            input_im = input_im.clamp(min=-1, max=1)
            gen_im_base = gen_im_base.clamp(min=-1, max=1)
            if args.use_counterfactual and cf_enabled_this_step:
                gen_im_keep = gen_im_keep.clamp(min=-1, max=1)

            loss = 0.0
            if g is not None:
                g_mean = g.float().mean().item()
                writter.add_scalar('Train/g_mean', g_mean, global_step)
                g_hist_values = g.detach().float().cpu().numpy()
                safe_add_histogram(
                    writter,
                    'Train/g_hist',
                    g_hist_values,
                    global_step,
                    fallback_values=g.detach().float().cpu()
                )
                g_total += g.numel()
                g_ones += g.sum().item()

            if args.loss_face_bg_weight:
                input_im_mask_hair, input_im_mask_face = parse_mask(args, input_im)
                input_im_bg_mask = ((input_im_mask_hair + input_im_mask_face) == 0).float()
                gen_im_mask_hair, gen_im_mask_face = parse_mask(args, gen_im_base)
                gen_im_bg_mask = ((gen_im_mask_hair + gen_im_mask_face) == 0).float()
                bg_mask = ((input_im_bg_mask + gen_im_bg_mask) == 2).float()

                loss_bg = torch.mean((input_im * bg_mask - gen_im_base * bg_mask) ** 2)
                loss = loss + loss_bg * args.loss_face_bg_weight
                bg_losses.update(loss_bg.item(), styles.size(0))
                writter.add_scalar('Train/Face BG loss', bg_losses.avg, global_step)

            if args.loss_id_weight:
                loss_id = args.id_loss(gen_im_base, input_im)
                loss = loss + loss_id * args.loss_id_weight
                id_losses.update(loss_id.item(), styles.size(0))
                writter.add_scalar('Train/ID loss', id_losses.avg, global_step)

            if args.loss_face_norm_weight:
                _, input_im_mask_face = parse_mask(args, input_im)
                _, gen_im_mask_face = parse_mask(args, gen_im_base)
                loss_face_norm = args.average_color_loss(gen_im_base, input_im, gen_im_mask_face, input_im_mask_face)
                loss = loss + loss_face_norm * args.loss_face_norm_weight
                face_norm_losses.update(loss_face_norm.item(), styles.size(0))
                writter.add_scalar('Train/Face norm loss', face_norm_losses.avg, global_step)

            if args.loss_w_norm_weight:
                loss_latent_norm = torch.mean(offset_base ** 2)
                loss = loss + loss_latent_norm * args.loss_w_norm_weight
                w_norm_losses.update(loss_latent_norm.item(), styles.size(0))
                writter.add_scalar('Train/W norm loss', w_norm_losses.avg, global_step)

            if args.loss_minmaxentropy_weight:
                off = offset_base.reshape(offset_base.size(0), -1).abs()
                offset_max = torch.max(off, 1)[0].unsqueeze(1)
                offset_min = torch.min(off, 1)[0].unsqueeze(1)
                offset_p = (off - offset_min) / (offset_max - offset_min + 1e-12) + 1e-7
                pseudo_entropy_loss = (-(offset_p * torch.log(offset_p)).sum(1).mean()) * 0.0001
                loss = loss + args.loss_minmaxentropy_weight * pseudo_entropy_loss
                entropy_losses.update(pseudo_entropy_loss.item(), styles.size(0))
                writter.add_scalar('Train/Entropy loss', entropy_losses.avg, global_step)

            if args.loss_clip_weight:
                loss_clip_base = clip_loss(gen_im_base, clip_text_base)
                loss_clip_base = torch.diag(loss_clip_base).mean()
                loss_clip = loss_clip_base
                if args.use_counterfactual and cf_enabled_this_step:
                    loss_clip_keep = clip_loss(gen_im_keep, clip_text_keep)
                    loss_clip_keep = torch.diag(loss_clip_keep).mean()
                    loss_clip = loss_clip + loss_clip_keep
                loss = loss + loss_clip * args.loss_clip_weight
                clip_losses.update(loss_clip.item(), styles.size(0))
                writter.add_scalar('Train/CLIP loss', clip_losses.avg, global_step)

            if args.use_adv and discriminator is not None and g is not None:
                adv_weight = args.adv_weight
                if args.adv_warmup_steps > 0:
                    warmup_factor = min(1.0, float(global_step + 1) / args.adv_warmup_steps)
                    adv_weight = adv_weight * warmup_factor
                writter.add_scalar('Train/adv_weight', adv_weight, global_step)

                logits_adv_base = discriminator(grad_reverse(delta_base_flat, args.grl_lambda))
                loss_adv_base = args.ce_criterion(logits_adv_base, g.long())
                loss_adv = loss_adv_base
                logits_for_acc = logits_adv_base
                g_for_acc = g
                if args.use_counterfactual and cf_enabled_this_step and args.adv_apply_to == 'both':
                    logits_adv_keep = discriminator(grad_reverse(delta_keep_flat, args.grl_lambda))
                    loss_adv_keep = args.ce_criterion(logits_adv_keep, g.long())
                    loss_adv = loss_adv + loss_adv_keep
                    logits_for_acc = torch.cat([logits_adv_base, logits_adv_keep], dim=0)
                    g_for_acc = torch.cat([g, g], dim=0)
                loss = loss + loss_adv * adv_weight
                adv_losses.update(loss_adv.item(), styles.size(0))
                writter.add_scalar('Train/loss_adv', adv_losses.avg, global_step)
                with torch.no_grad():
                    d_acc = (logits_for_acc.argmax(dim=1) == g_for_acc).float().mean().item()
                writter.add_scalar('Train/D_acc', d_acc, global_step)

            if args.use_counterfactual and g is not None and cf_enabled_this_step:
                keep_attr = args.face_model(torchvision.transforms.functional.resize(gen_im_keep, 256))
                logits_keep = torch.stack(keep_attr).transpose(0, 1)
                protected_logits = logits_keep[:, args.protect_attr_idx, :]
                loss_keep = args.ce_criterion(protected_logits, g.long())
                loss = loss + loss_keep * args.w_keep
                keep_losses.update(loss_keep.item(), styles.size(0))
                writter.add_scalar('Train/L_keep', keep_losses.avg, global_step)
                with torch.no_grad():
                    keep_preds = logits_keep.argmax(-1)
                    protect_flip_rate = (keep_preds[:, args.protect_attr_idx] != g).float().mean().item()
                protect_flip_rates.update(protect_flip_rate, styles.size(0))
                writter.add_scalar('Train/protect_flip_rate', protect_flip_rates.avg, global_step)
            elif args.use_counterfactual and g is not None and not cf_enabled_this_step:
                keep_losses.update(0.0, styles.size(0))
                writter.add_scalar('Train/L_keep', keep_losses.avg, global_step)

            all_losses.update(loss.item(), styles.size(0))
            writter.add_scalar('Train/all loss', all_losses.avg, global_step)

            scale = float(mb_count) / float(batch_size)
            (loss * scale).backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            vis_ = make_grid(gen_im_base[:9].clamp(min=-1, max=1) * 0.5 + 0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'out_face')
            os.makedirs(save_path, exist_ok=True)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch) + '.png'))

            vis_ = make_grid(input_im[:9].clamp(min=-1, max=1) * 0.5 + 0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'in_face')
            os.makedirs(save_path, exist_ok=True)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch) + '.png'))

            save_path = os.path.join(args.save_folder, 'text')
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, str(epoch) + '.txt'), 'w') as n:
                safe_write_text_lines(n, sampled_text, max_lines=9)

    if g_total > 0:
        g_zeros = g_total - g_ones
        print(
            f"Epoch {epoch} g distribution (protect_attr_idx={args.protect_attr_idx}): "
            f"0={g_zeros} 1={g_ones} mean={g_ones / g_total:.4f}"
        )

    return clip_losses.avg


def validate(eval_loader, model, writter, generator, clip_loss, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    clip_losses = AverageMeter('clip_loss', ':.4e')
    w_norm_losses = AverageMeter('w_norm_loss', ':.4e')
    bg_losses = AverageMeter('bg_loss', ':.4e')
    id_losses = AverageMeter('id_loss', ':.4e')
    face_norm_losses = AverageMeter('face_norm_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    all_losses = AverageMeter('all_losses', ':.4e')

    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, data_time, all_losses, clip_losses, w_norm_losses, id_losses, face_norm_losses, entropy_losses],
        prefix=f"Epoch: [{epoch}]"
    )

    model.eval()
    acc_avg = AverageMeter()

    features = []
    end = time.time()

    log_g = args.log_g_only or args.use_state_mod or args.use_adv or args.use_counterfactual
    g_total = 0
    g_ones = 0
    g_values = []
    for i, (clip_text, sampled_text, labels, exist_mask, length, test_latents) in enumerate(eval_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            clip_text = clip_text.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            exist_mask = exist_mask.cuda(args.gpu, non_blocking=True)
            test_latents = test_latents.cuda(args.gpu, non_blocking=True)

        code = test_latents
        styles = generator.style(code)
        input_im, _ = generator([styles], input_is_latent=True, randomize_noise=False,
                                truncation=args.truncation, truncation_latent=args.mean_latent)

        g = None
        in_attr = None
        if log_g:
            in_attr = args.face_model(torchvision.transforms.functional.resize(input_im, 256))
            in_preds = torch.stack(in_attr).transpose(0, 1).argmax(-1)  # [B, 40]
            g = in_preds[:, args.protect_attr_idx]  # [B]
            g_total += g.numel()
            g_ones += g.sum().item()
            g_values.append(g.detach().to('cpu'))

        offset = model(styles, clip_text, g if args.use_state_mod else None)
        new_styles = styles.unsqueeze(1).repeat(1, 14, 1) + offset

        gen_im, _ = generator([new_styles], input_is_latent=True, randomize_noise=False,
                              truncation=args.truncation, truncation_latent=args.mean_latent)

        input_im = input_im.clamp(min=-1, max=1)
        gen_im = gen_im.clamp(min=-1, max=1)

        if in_attr is None:
            in_attr = args.face_model(torchvision.transforms.functional.resize(input_im, 256))
        gen_attr = args.face_model(torchvision.transforms.functional.resize(gen_im, 256))
        in_preds = torch.stack(in_attr).transpose(0, 1).argmax(-1)
        gen_preds = torch.stack(gen_attr).transpose(0, 1).argmax(-1)
        out_label = torch.where(exist_mask == 1, labels.long(), in_preds)
        acc = (((gen_preds == out_label).sum(1) / gen_preds.size(1)).mean().item()) * 100
        acc_avg.update(acc, styles.size(0))

        feat = args.inception(gen_im)[0].view(gen_im.shape[0], -1)
        features.append(feat.to('cpu'))

        loss = 0.0

        if args.loss_face_bg_weight:
            input_im_mask_hair, input_im_mask_face = parse_mask(args, input_im)
            input_im_bg_mask = ((input_im_mask_hair + input_im_mask_face) == 0).float()
            gen_im_mask_hair, gen_im_mask_face = parse_mask(args, gen_im)
            gen_im_bg_mask = ((gen_im_mask_hair + gen_im_mask_face) == 0).float()
            bg_mask = ((input_im_bg_mask + gen_im_bg_mask) == 2).float()

            loss_bg = torch.mean((input_im * bg_mask - gen_im * bg_mask) ** 2)
            loss = loss + loss_bg * args.loss_face_bg_weight
            bg_losses.update(loss_bg.item(), styles.size(0))
            if i == len(eval_loader) - 1 and writter is not None:
                writter.add_scalar('Val/Face BG loss', bg_losses.avg * 100, epoch)

        if args.loss_id_weight:
            loss_id = args.id_loss(gen_im, input_im)
            loss = loss + loss_id * args.loss_id_weight
            id_losses.update(loss_id.item(), styles.size(0))
            if i == len(eval_loader) - 1 and writter is not None:
                writter.add_scalar('Val/ID loss', id_losses.avg * 100, epoch)

        if args.loss_face_norm_weight:
            _, input_im_mask_face = parse_mask(args, input_im)
            _, gen_im_mask_face = parse_mask(args, gen_im)
            loss_face_norm = args.average_color_loss(gen_im, input_im, gen_im_mask_face, input_im_mask_face)
            loss = loss + loss_face_norm * args.loss_face_norm_weight
            face_norm_losses.update(loss_face_norm.item(), styles.size(0))
            if i == len(eval_loader) - 1 and writter is not None:
                writter.add_scalar('Val/Face norm loss', face_norm_losses.avg * 100, epoch)

        if args.loss_w_norm_weight:
            loss_latent_norm = torch.mean(offset ** 2)
            loss = loss + loss_latent_norm * args.loss_w_norm_weight
            w_norm_losses.update(loss_latent_norm.item(), styles.size(0))
            if i == len(eval_loader) - 1 and writter is not None:
                writter.add_scalar('Val/W norm loss', w_norm_losses.avg * 100, epoch)

        if args.loss_minmaxentropy_weight:
            off = offset.reshape(offset.size(0), -1).abs()
            offset_max = torch.max(off, 1)[0].unsqueeze(1)
            offset_min = torch.min(off, 1)[0].unsqueeze(1)
            offset_p = (off - offset_min) / (offset_max - offset_min + 1e-12) + 1e-7
            pseudo_entropy_loss = (-(offset_p * torch.log(offset_p)).sum(1).mean()) * 0.0001
            loss = loss + args.loss_minmaxentropy_weight * pseudo_entropy_loss
            entropy_losses.update(pseudo_entropy_loss.item(), styles.size(0))
            if i == len(eval_loader) - 1 and writter is not None:
                writter.add_scalar('Val/Entropy loss', entropy_losses.avg, epoch)

        if args.loss_clip_weight:
            loss_clip = clip_loss(gen_im, clip_text)
            loss_clip = torch.diag(loss_clip).mean()
            loss = loss + loss_clip * args.loss_clip_weight
            clip_losses.update(loss_clip.item(), styles.size(0))
            if i == len(eval_loader) - 1 and writter is not None:
                writter.add_scalar('Val/CLIP loss', clip_losses.avg * 100, epoch)

        all_losses.update(loss.item(), styles.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            vis_ = make_grid(gen_im[:9].clamp(min=-1, max=1) * 0.5 + 0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'eval_out_face')
            os.makedirs(save_path, exist_ok=True)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch) + '.png'))

            vis_ = make_grid(input_im[:9].clamp(min=-1, max=1) * 0.5 + 0.5, nrow=3, normalize=False)
            save_path = os.path.join(args.save_folder, 'eval_in_face')
            os.makedirs(save_path, exist_ok=True)
            torchvision.utils.save_image(vis_, os.path.join(save_path, str(epoch) + '.png'))

            save_path = os.path.join(args.save_folder, 'eval_text')
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, str(epoch) + '.txt'), 'w') as n:
                safe_write_text_lines(n, sampled_text, max_lines=9)

    if len(features) == 0:
        raise RuntimeError("No features extracted in validate(). Check eval_loader / dataset length.")

    features = torch.cat(features, 0).numpy()
    print(f'extracted {features.shape[0]} features')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)
    if sample_cov.ndim == 0:
        sample_cov = np.zeros((features.shape[1], features.shape[1]))

    fid = calc_fid(sample_mean, sample_cov, args.real_mean, args.real_cov)
    if writter is not None:
        writter.add_scalar('Val/all loss', all_losses.avg, epoch)
        writter.add_scalar('Val/fid', fid, epoch)
        writter.add_scalar('Val/face acc', acc_avg.avg, epoch)
        writter.add_scalar('Val/face id sim', 100 * (1 - id_losses.avg), epoch)
        if log_g and g_total > 0:
            writter.add_scalar('Val/g_mean', g_ones / g_total, epoch)
            if g_values:
                g_hist_values = torch.cat(g_values, dim=0).float().cpu().numpy()
                safe_add_histogram(
                    writter,
                    'Val/g_hist',
                    g_hist_values,
                    epoch,
                    fallback_values=torch.from_numpy(g_hist_values)
                )

    print()
    print(f'fid: {fid}', flush=True)
    print(f'face id sim: {100 * (1 - id_losses.avg)}', flush=True)
    print(f'face attribute accuracy: {acc_avg.avg}', flush=True)

    return fid


class TextDataset(data.Dataset):
    def __init__(self, split='train', train_num=None, val_num=None):
        self.text_dir = 'data/celeba-caption/'
        self.text_files = os.listdir(self.text_dir)
        self.text_files.sort(key=int_item)

        f = open('data/list_attr_celeba.txt')
        data_lines = f.readlines()
        attrs = data_lines[1].split(' ')
        attrs[-1] = attrs[-1][:-1]
        self.attrs = np.array([' '.join(a.split('_')).lower() for a in attrs], dtype=object)
        self.anno = data_lines[2:]

        if train_num is None:
            train_num = 25000
        if split == 'train':
            self.text_files = self.text_files[:train_num]
            self.anno = self.anno[:train_num]
        else:
            self.test_latents = torch.load('data/test_latents_seed100.pt')
            if val_num is None:
                val_num = self.test_latents.shape[0] - train_num
            val_num = min(val_num, self.test_latents.shape[0])
            self.text_files = self.text_files[train_num:train_num + val_num]
            self.anno = self.anno[train_num:train_num + val_num]

        self.split = split
        self.non_represents = ['no', 'hair', 'wearing', 'eyebrows', 'eyes', 'big', 'nose', 'o']
        self.gender_list = ['he', 'she', 'man', 'woman']

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, index):
        text_filename = self.text_files[index]
        text_path = os.path.join(self.text_dir, text_filename)
        text_set = open(text_path).readlines()

        sampled_text = text_set[0][:-1]
        anno = self.anno[index][:-1].split(' ')[1:]
        clip_text, labels, exist_mask = produce_labels(sampled_text, anno, self.attrs, self.gender_list, self.non_represents)

        length = torch.where(clip_text == 0)[1][0].item()

        if self.split == 'train':
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length
        else:
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length, self.test_latents[index]


class PartTextDataset(data.Dataset):
    def __init__(self, split='train', sample_num=3, train_num=None, val_num=None):
        self.test_latents = torch.load('data/test_latents_seed100.pt')
        self.split = split
        self.sample_num = sample_num

        f = open('data/list_attr_celeba.txt')
        self.data = f.readlines()
        attrs = self.data[1].split(' ')
        attrs[-1] = attrs[-1][:-1]
        self.attrs = np.array([' '.join(a.split('_')).lower() for a in attrs], dtype=object)

        if train_num is None:
            train_num = 25000
        if split == 'train':
            self.img_attr = self.data[2:2 + train_num]
        else:
            if val_num is None:
                val_num = self.test_latents.shape[0] - train_num
            val_num = min(val_num, self.test_latents.shape[0])
            start = 2 + train_num
            self.img_attr = self.data[start:start + val_num]

        self.hair = ['bald', 'bangs', 'black hair', 'blond hair', 'brown hair', 'gray hair',
                     'receding hairline', 'straight hair', 'wavy hair']
        self.eye = ['arched eyebrows', 'bags under eyes', 'bushy eyebrows', 'eyeglasses', 'narrow eyes']
        self.fashion = ['attractive', 'heavy makeup', 'high cheekbones', 'rosy cheeks', 'wearing earrings',
                        'wearing hat', 'wearing lipstick', 'wearing necklace', 'wearing necktie']
        self.others = ['5 o clock shadow', 'big nose', 'blurry', 'chubby', 'double chin', 'no beard',
                       'oval face', 'pale skin', 'pointy nose', 'young']
        self.mouth = ['big lips', 'mouth slightly open', 'smiling', 'goatee', 'mustache', 'sideburns']

        self.groups = [self.hair, self.eye, self.fashion, self.others, self.mouth]

    def __len__(self):
        return len(self.img_attr)

    def __getitem__(self, index):
        sampled_class = torch.randint(0, 5, (1,)).item()
        instance_attr = self.groups[sampled_class]
        sampled_cate = torch.randperm(len(instance_attr))[:self.sample_num]
        attr = np.array(instance_attr)[sampled_cate]
        if self.sample_num == 1:
            attr = np.array([attr])

        # FIX: numpy.int64 -> Python int, then index as torch.long
        selected_cate_40 = []
        for x in attr:
            selected_cate_40.append(int(np.where(self.attrs == x)[0][0]))

        gender = torch.randint(0, 3, (1,)).item()
        concat_text = ', '.join(attr)
        if gender == 0:
            sampled_text = 'she has ' + concat_text
        elif gender == 1:
            sampled_text = 'he has ' + concat_text
        else:
            sampled_text = 'the person has ' + concat_text

        clip_text = clip.tokenize(sampled_text)

        exist_mask = torch.zeros(40, dtype=torch.float32)
        idx = torch.tensor(selected_cate_40, dtype=torch.long)
        exist_mask[idx] = 1.0
        labels = exist_mask.clone()

        if gender != 2:
            exist_mask[20] = 1
            if gender == 1:
                labels[20] = 1
            else:
                labels[20] = 0

        length = torch.where(clip_text == 0)[1][0].item()

        if self.split == 'train':
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length
        else:
            return clip_text.squeeze(0), sampled_text, labels, exist_mask, length, self.test_latents[index]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StateModulator(nn.Module):
    def __init__(self, in_dim: int = 512, out_layers: int = 14, bias_init: float = 2.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim + 1, in_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_layers)
        self.bias = nn.Parameter(torch.full((out_layers,), bias_init))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, text_embedding: Tensor, g: Tensor) -> Tensor:
        # text_embedding: [B, 512], g: [B] -> s: [B, 14, 1]
        g = g.float().unsqueeze(1)
        x = torch.cat([text_embedding, g], dim=1)
        raw = self.fc2(self.act(self.fc1(x)))
        s = torch.sigmoid(raw + self.bias)
        return s.unsqueeze(-1)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(x: Tensor, lambda_: float) -> Tensor:
    return GradientReversal.apply(x, lambda_)


class SubgroupDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransModel(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 4, dim_feedforward: int = 2048,
                 activation: str = "relu", dropout: float = 0.2,
                 num_decoder_layers: int = 4):
        super(TransModel, self).__init__()

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
        self.face_pool = nn.AdaptiveAvgPool2d((224, 224))

        self.pos_encoder = PositionalEncoding(d_model, max_len=20)
        self.x_map = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.text_map = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.state_modulator = StateModulator(in_dim=d_model, out_layers=14, bias_init=2.5)
        self.last_s = None

    def forward(self, x, text_inputs, g: Optional[Tensor] = None):
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs).detach()  # [B, 512]
        text_embedding = self.text_map(text_embedding)
        text_embedding = text_embedding + self.text_map(text_embedding)
        text_embedding = self.norm1(text_embedding)

        x = x.unsqueeze(1).repeat(1, 14, 1).transpose(0, 1)  # [14, B, 512]
        x = self.pos_encoder(x)
        x = self.x_map(x)
        x = x + self.x_map(x)
        x = self.norm2(x)

        out = self.decoder(tgt=x, memory=text_embedding.unsqueeze(1).transpose(0, 1))
        out = out.transpose(0, 1)  # [B, 14, 512]
        if g is not None:
            s = self.state_modulator(text_embedding, g)  # [B, 14, 1]
            out = out * s
            self.last_s = s
        else:
            self.last_s = None
        return out


if __name__ == "__main__":
    main()
