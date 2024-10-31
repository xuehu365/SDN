import os
from PIL import Image
import warnings
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import bisect
from sampler import Random_Balanced_sampler
from torchvision import transforms
import torch


class ResizeImage:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

image_train = transforms.Compose([
    ResizeImage(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_freq_train = transforms.Compose([
    ResizeImage(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

image_test = transforms.Compose([
    ResizeImage(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

rand_augmentation = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(1, 2.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PlaceCrop:

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ConcatDataset(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn('cummulative_sizes attribute is renamed to '
                      'cumulative_sizes', DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class ImageList(Dataset):
    def __init__(self, image_root, image_list_root, domain_name, domain_label, dataset_name, split='train', transform=None,
                 strong_transform=None, aug_num=0, rand_aug=False, freq=False):
        self.image_root = image_root
        self.domain_name = domain_name
        self.dataset_name = dataset_name
        self.transform = transform
        self.strong_transform = strong_transform
        self.loader = self._rgb_loader
        self.rand_aug = rand_aug
        self.aug_num = aug_num
        self.freq = freq

        if dataset_name == 'domainnet':
            imgs = self._make_dataset(os.path.join(image_list_root, domain_name + '_' + split + '.txt'), domain_label)
        else:
            imgs = self._make_dataset(os.path.join(image_list_root, domain_name + '.txt'), domain_label)
        
        self.imgs = imgs
        self.tgts = [s[1] for s in imgs]

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _make_dataset(self, image_list_path, domain):
        print('image path', image_list_path)
        image_list = open(image_list_path).readlines()
        images = [(val.split()[0], int(val.split()[1]), int(domain)) for val in image_list]
        return images

    def __getitem__(self, index):
        output = {}
        path, _, domain = self.imgs[index]

        raw_img = self.loader(os.path.join(self.image_root, path))

        if self.transform is not None and self.freq == False:
            img = self.transform(raw_img)
        elif self.freq == True:
            img = image_freq_train(raw_img)
        
        if self.rand_aug and self.strong_transform != None:
            aug_img = [self.strong_transform(raw_img) for i in range(self.aug_num)]
            output['strong_img'] = aug_img

        output['img'] = img
        output['target'] = torch.squeeze(torch.LongTensor([np.int64(self.tgts[index]).item()]))
        output['domain'] = domain
        output['idx'] = index

        return output

    def __len__(self):
        return len(self.imgs)


def build_dataset(args, dataset_name, source_index, bs, num_workers):
    all_domains = {
        'office31':
            {
                'path': 'SDN/data/office31/',
                'list_root': 'SDN/data/office31/',
                'sub_domains': ['amazon', 'dslr', 'webcam'],
                'numbers': [2817, 498, 795],
                'classes': 31
            },
        'office-home':
            {
                'path': 'SDN/Dataset/Office-Home/',
                'list_root': 'SDN/Dataset/Office-Home/',
                'sub_domains': ['Art', 'Clipart', 'Product', 'Real_World'],
                'numbers': [2427, 4365, 4439, 4357],
                'classes': 65
            },
        'imageCLEF':
            {
                'path': 'SDN/data/imageCLEF/',
                'list_root': 'SDN/data/imageCLEF/',
                'sub_domains': ['b', 'c', 'i', 'p'],
                'numbers': [600, 600, 600, 600],
                'classes': 12
            },
        'domainnet':
            {
                'path': 'SDN/data/domainnet/',
                'list_root': 'SDN/data/domainnet/',
                'sub_domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                'numbers': [14604, 15582, 21850, 51750, 52041, 20916],
                'classes': 345
            }
    }

    args.num_classes = all_domains[dataset_name]['classes']
    args.num_domains = len(all_domains[dataset_name]['sub_domains'])

    dsets = {
        'target_train': {},
        'target_test': {},
    }
    dset_loaders = {
        'target_train': {},
        'target_test': {},
    }
    samplers = {}
    concate_sets = {}
    concate_loaders = {}


    dsets['source'] = ImageList(image_root=all_domains[dataset_name]['path'],
                                image_list_root=all_domains[dataset_name]['list_root'],
                                domain_name=all_domains[dataset_name]['sub_domains'][source_index],
                                transform=image_train,
                                domain_label=source_index,
                                dataset_name=dataset_name, split='train',
                                strong_transform=rand_augmentation,
                                aug_num=args.aug_num,
                                rand_aug=args.rand_aug,
                                freq=args.freq)

    dsets['source_test'] = ImageList(image_root=all_domains[dataset_name]['path'],
                                     image_list_root=all_domains[dataset_name]['list_root'],
                                     domain_name=all_domains[dataset_name]['sub_domains'][source_index],
                                     transform=image_test,
                                     domain_label=source_index,
                                     dataset_name=dataset_name,
                                     split='test',
                                     freq=False)

    if all_domains[dataset_name]['classes'] * bs > args.bs_limit:
        scale_bs = int(args.bs_limit / bs) * bs
    else:
        scale_bs = all_domains[dataset_name]['classes'] * bs

    samplers['source'] = Random_Balanced_sampler(dsets['source'], all_domains[dataset_name]['classes'], bs, args.bs_limit)
    dset_loaders['source'] = DataLoader(dsets['source'],
                                            num_workers=num_workers,
                                            pin_memory=False,
                                            batch_sampler=samplers['source'])
   
    dset_loaders['source_test'] = DataLoader(dataset=dsets['source_test'],
                                             batch_size=64,
                                             num_workers=num_workers,
                                             drop_last=False,
                                             pin_memory=False)

    for i in range(args.num_domains):
        if i == source_index:
            continue

        dset_name = all_domains[dataset_name]['sub_domains'][i]

        dsets['target_train'][dset_name] = ImageList(image_root=all_domains[dataset_name]['path'],
                                                     image_list_root=all_domains[dataset_name]['list_root'],
                                                     domain_name=dset_name,
                                                     transform=image_train,
                                                     domain_label=i,
                                                     dataset_name=dataset_name,
                                                     split='train',
                                                     strong_transform=rand_augmentation,
                                                     aug_num=args.aug_num,
                                                     rand_aug=args.rand_aug,
                                                     freq=args.freq)

        dsets['target_test'][dset_name] = ImageList(image_root=all_domains[dataset_name]['path'],
                                                    image_list_root=all_domains[dataset_name]['list_root'],
                                                    domain_name=dset_name,
                                                    transform=image_test,
                                                    domain_label=i,
                                                    dataset_name=dataset_name,
                                                    split='test',
                                                    freq=False)

        dset_loaders['target_train'][dset_name] = DataLoader(dataset=dsets['target_train'][dset_name],
                                                             batch_size=scale_bs,
                                                             shuffle=True,
                                                             num_workers=num_workers,
                                                             drop_last=True,
                                                             pin_memory=False)

        dset_loaders['target_test'][dset_name] = DataLoader(dataset=dsets['target_test'][dset_name],
                                                            batch_size=64,
                                                            num_workers=num_workers,
                                                            drop_last=False,
                                                            pin_memory=False)

    concate_sets['train'] = ConcatDataset(dsets['target_train'].values())
    concate_sets['test'] = ConcatDataset(dsets['target_test'].values())
    concate_loaders['train'] = DataLoader(dataset=concate_sets['train'],
                                          batch_size=scale_bs,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          drop_last=True,
                                          pin_memory=False)
    concate_loaders['test'] = DataLoader(dataset=concate_sets['test'],
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         drop_last=True,
                                         pin_memory=False)

    return dsets, dset_loaders, concate_sets, concate_loaders
