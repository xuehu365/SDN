import sys
path = ''
sys.path.append(path)
import argparse
import torch.optim as optim
from preparedata_lds import build_dataset
from models.lr_scheduler import StepwiseLR
from models.resnet_lowlevel import *
from models.discriminator import *
from models.grl import *
from util import *
import copy
import os
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from models.Loss import AdversarialLoss_PDD

def cal_acc(domain_name, encoder, test_data, logger=None):
    encoder.eval()
    loss, acc = 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for image_folder in tqdm(test_data):
            images = image_folder['img']
            labels = image_folder['target']
            images = images.cuda()
            labels = labels.cuda()
            _, preds = encoder(images)
            loss += criterion(preds, labels).item()
            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(test_data)
    acc /= len(test_data.dataset)
    logger.info("{}: Avg Loss = {}, Avg Accuracy = {:2%}".format(domain_name, loss, acc))
    return acc


def main(args):
    dsets, dset_loaders, _, concate_loaders = build_dataset(args=args,
                                                            dataset_name=args.dataset,
                                                            source_index=args.source,
                                                            bs=args.batch_size,
                                                            num_workers=args.workers)
    dset_loaders_source_test = copy.deepcopy(dset_loaders['source'])
    dset_loaders['source'] = ForeverDataIterator(dset_loaders['source'])
    concate_loaders['train'] = ForeverDataIterator(concate_loaders['train'])
    for t in dset_loaders['target_train'].keys():
        dset_loaders['target_train'][t] = ForeverDataIterator(dset_loaders['target_train'][t])
    if args.net == 'resnet50':
        encoder = resnet50(args, pretrained=True).cuda()
    elif args.net == 'resnet101':
        encoder = resnet101(args, pretrained=True).cuda()
    discriminator = CatDomain(args.feat_dim, args.hid_dim, args.num_classes).cuda()
    PDD_adv = AdversarialLoss_PDD(encoder.fc2).cuda()

    logger_path = path + 'logs/{}/{}_{}.log'.format(args.dataset, dsets['source'].domain_name, args.sub_log)
    logger = get_logger(logger_path)
    logger.info(args)
    logger.info('Start training process')

    resent_params = [
        {'params': encoder.conv1.parameters(), 'name': 'conv', "lr_mult": 0.1},
        {'params': encoder.bn1.parameters(), 'name': 'conv', "lr_mult": 0.1},
        {'params': encoder.layer1.parameters(), 'name': 'conv', "lr_mult": 0.1},
        {'params': encoder.layer2.parameters(), 'name': 'conv', "lr_mult": 0.1},
        {'params': encoder.layer3.parameters(), 'name': 'conv', "lr_mult": 0.1},
        {'params': encoder.layer4.parameters(), 'name': 'conv', "lr_mult": 0.1},
        {'params': encoder.fc1.parameters(), 'name': 'ca_cl', "lr_mult": 1.0},
        {'params': encoder.fc2.parameters(), 'name': 'ca_cl', "lr_mult": 1.0},
    ]
    optim_g = optim.SGD(resent_params + discriminator.get_parameters(), args.lr, momentum=0.9, weight_decay=0.001, nesterov=True)
    lr_scheduler = StepwiseLR(optim_g, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    discriminator.train()
    cls_criterion = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(reduction='none')
    grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
    best_acc, s_acc, t_acc = 0.0, 0.1, 0.0
    cls_loss = torch.randn(1)
    loss = torch.randn(1)
    d_loss = torch.randn(1)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.max_epoch):
        encoder.train()
        PDD_adv.train()
        lr_scheduler.step()

        total_filtered = 0
        filtered_stats = torch.zeros(args.num_classes)
        filtered_true_stats = torch.zeros(args.num_classes)

        max_iters = args.iter_epoch * args.max_epoch
        for step in range(args.iter_epoch):
            current_iter = step + args.iter_epoch * epoch
            rho = current_iter / max_iters
            s_data, t_concate_data = next(dset_loaders['source']), next(concate_loaders['train'])
            s_img = s_data['img'].cuda()
            t_concate_img = t_concate_data['img'].cuda()
            s_label_origin = s_data['target']
            t_concate_label_origin = t_concate_data['target']
            s_label = to_onehot(s_label_origin, args.num_classes).cuda()
            s_label_origin = s_label_origin.cuda()
            optim_g.zero_grad()

            if args.amp:
                with autocast():
                    imgs = torch.cat((s_img, t_concate_img), dim=0)
                    f, pred = encoder(imgs)
                    f_s, f_st, f_t = f.chunk(3, dim=0)
                    pred_s, pred_st, pred_t = pred.chunk(3, dim=0)
                    cls_loss = cls_criterion(pred_s, s_label_origin) * 0.5 + cls_criterion(pred_st, s_label_origin) * 0.5
                    f_s_t = torch.cat((f_s, f_t), dim=0)
                    pdd_loss = PDD_adv(f_s_t, s_label_origin)
                    pseudo_label = torch.softmax(pred_t, dim=1).detach()
                    entropy = torch.sum(-torch.log(pseudo_label + 1e-16) * pseudo_label, dim=1).cuda()
                    filtered = torch.sum(entropy < args.margin)
                    total_filtered += filtered
                    if filtered > 0:
                        _, ind = torch.max(pseudo_label[entropy < args.margin, :], dim=1)
                        pseudo_label[entropy < args.margin, :] = 0
                        pseudo_label[entropy < args.margin, ind] = 1
                        for kk in ind:
                            filtered_stats[kk] += 1
                        for kk in t_concate_label_origin[entropy < args.margin]:
                            filtered_true_stats[kk] += 1

                    ff = grl(torch.cat((f_s, f_t), dim=0))
                    d_pred = discriminator(ff)
                    d_pred_s, d_pred_t = d_pred.chunk(2, dim=0)
                    d_label_s = torch.ones((f_s.size(0), args.num_classes)).to(f_s.device)
                    d_label_t = torch.zeros((f_t.size(0), args.num_classes)).to(f_t.device)
                    d_loss_s = bce(d_pred_s, d_label_s) * s_label * 0.5
                    d_loss_t = bce(d_pred_t, d_label_t) * pseudo_label * 0.5
                    d_loss = torch.mean(torch.sum((d_loss_s + d_loss_t), dim=1))

                    loss = cls_loss + d_loss - args.pdd_tradeoff * rho * pdd_loss

                    scaler.scale(loss).backward()
                    scaler.step(optim_g)
                    scaler.update()
            
            if (step + 1) % 10 == 0:
                logger.info(
                    "Epoch [{}/{}] Step [{}/{}]: loss={:.3f} d_loss={:.3f} cls_loss={:.3f} s_acc={:.4f} t_acc={:.4f} best={:.4f}"
                    .format(epoch + 1,
                            args.max_epoch,
                            step + 1,
                            args.iter_epoch,
                            loss.item(),
                            -d_loss.item(),
                            cls_loss.item(),
                            s_acc,
                            t_acc,
                            best_acc)
                )

        if (epoch + 1) % args.eval_epoch == 0:
            logger.info('Total filtered:{}'.format(total_filtered))
            logger.info('Filtered class:{}'.format(filtered_stats))
            logger.info('Filtered true:{}'.format(filtered_true_stats))

            s_acc = cal_acc(dsets['source'].domain_name, encoder, dset_loaders_source_test, logger)

            t_acc = []
            for t in dset_loaders['target_test'].keys():
                t_acc.append(cal_acc(t, encoder, dset_loaders['target_test'][t], logger))
            t_acc = torch.Tensor(t_acc).mean().item()

            logger.info('t_acc_mean:{}'.format(t_acc))


        if t_acc > best_acc:
            best_acc = t_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SDN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--dataset', type=str, default='office31', choices=['office31', 'office-home', 'imageCLEF', 'domainnet'])
    parser.add_argument('--source', type=int, default=0, help="source dataset subclass")
    parser.add_argument('--max_epoch', type=int, default=20, help="max epochs")
    parser.add_argument('--eval_epoch', type=int, default=1, help="evaluation epoch")
    parser.add_argument('--iter_epoch', type=int, default=30, help="iterations per epoch")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--feat_dim', type=int, default=1024, help="feature space dimension")
    parser.add_argument('--hid_dim', type=int, default=2048, help="feature space dimension")
    parser.add_argument('--bs_limit', type=int, default=32, help="maximum batch size limit due to GPU mem")
    parser.add_argument('--workers', type=int, default=0, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--margin', type=float, default=0.05,help='marign for filetering the pseudo labels')
    parser.add_argument('--rand_aug', action='store_true', help='whether to align categorical domain')
    parser.add_argument('--aug_num', type=int, default=1, help="source dataset subclass")
    parser.add_argument('--sub_log', type=str, default='250', help="sub log number")
    parser.add_argument('--amp', type=bool, default='True', help="whether to use amp fp16 for training")
    parser.add_argument('--freq', action='store_true', help="whether to fourier augmentation")
    parser.add_argument('--pdd_tradeoff', type=float, default=1.0, help="hyper-parameter: alpha0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    init_random_seed(args.seed)

    main(args)
