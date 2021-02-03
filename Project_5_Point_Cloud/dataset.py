import os
import os.path as osp
import glob

import math

import zipfile

import urllib

import shutil

import torch


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def parse_off(src):
    if src[0] == 'OFF':
        src = src[1:]
    else:
        src[0] = src[0][3:]

    num_nodes, num_faces = [int(item) for item in src[0].split()[:2]]

    pos = parse_txt_array(src[1:1 + num_nodes])

    face = src[1 + num_nodes:1 + num_nodes + num_faces]
    face = face_to_tri(face)

    return pos, face


def face_to_tri(face):
    face = [[int(x) for x in line.strip().split(' ')] for line in face]

    triangle = torch.tensor([line[1:] for line in face if line[0] == 3])
    triangle = triangle.to(torch.int64)

    rect = torch.tensor([line[1:] for line in face if line[0] == 4])
    rect = rect.to(torch.int64)

    if rect.numel() > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [1, 2, 3]]
        return torch.cat([triangle, first, second], dim=0).t().contiguous()
    else:
        return triangle.t().contiguous()


def read_off(path):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_off(src)


def sample_points(pos, face, num=1024):
    assert pos.size(1) == 3 and face.size(0) == 3

    pos_max = pos.max()
    pos = pos / pos_max

    area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    sample = torch.multinomial(prob, num, replacement=True)
    face = face[:, sample]

    frac = torch.rand(num, 2, device=pos.device)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]

    pos_sampled = pos[face[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    pos_sampled = pos_sampled * pos_max

    return pos_sampled


class ModelNet(torch.utils.data.Dataset):
    url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'

    def __init__(self, root, train=True, transform=None):
        super(ModelNet, self).__init__()
        self.root = root
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        self.transform = transform

        self.download()
        self.process()

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.targets = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos, target = self.data[idx], int(self.targets[idx])
        if self.transform is not None:
            pos = self.transform(pos)
        return pos, target

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def raw_paths(self):
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        files = self.processed_file_names
        return [osp.join(self.processed_dir, f) for f in files]

    @property
    def num_classes(self):
        return int(self.targets.max().item()) + 1 

    def download(self):

        if all([osp.exists(f) for f in self.raw_paths]):
            return

        os.makedirs(osp.expanduser(osp.normpath(self.raw_dir)))

        filename = self.url.rpartition('/')[2]
        path = osp.join(self.root, filename)
        if osp.exists(path):
            print('Using exist file', filename)
        else:
            print('Downloading', self.url)
            data = urllib.request.urlopen(self.url)
            with open(path, 'wb') as f:
                f.write(data.read())

        with zipfile.ZipFile(path, 'r') as f:
            print('Extracting', path)
            f.extractall(self.root)
        os.unlink(path)

        folder = osp.join(self.root, 'ModelNet10')
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)
        print('Done!')

    def process(self):
        if all([osp.exists(f) for f in self.processed_paths]):
            return

        print('Processing...')
        os.makedirs(osp.expanduser(osp.normpath(self.processed_dir)))

        self.process_set('train', self.processed_paths[0])
        self.process_set('test', self.processed_paths[1])

        print('Done!')

    def process_set(self, dataset, processed_path):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        positions = []
        targets = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))

            for path in paths:
                pos, face = read_off(path)

                scale = (1 / pos.abs().max()) * 0.999999
                pos = pos * scale

                pos = sample_points(pos, face)
                positions.append(pos.t())
                targets.append(target)

        positions = torch.stack(positions)
        targets = torch.Tensor(targets)

        torch.save((positions, targets), processed_path)


def fixed_points(pos, y, num):
    N, D = pos.shape
    assert D == 3
    choice = torch.cat([torch.randperm(N)
                        for _ in range(math.ceil(num / N))], dim=0)[:num]
    return pos[choice], y[choice]

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ShapeNet(torch.utils.data.Dataset):
    url = 'https://shapenet.cs.stanford.edu/iccv17/partseg'

    categories = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    def __init__(self, root, category, train=True, transform=None):
        super(ShapeNet, self).__init__()
        self.category = category

        assert self.category in self.categories

        self.root = root
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        self.transform = transform

        self.download()
        self.process()

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.targets = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            pos = self.transform(pos)
        return pos, target

    @property
    def raw_file_names(self):
        return [
            'train_data', 'train_label', 'val_data', 'val_label', 'test_data',
            'test_label'
        ]

    @property
    def processed_file_names(self):
        names = ['training.pt', 'test.pt']
        return [osp.join(self.category, name) for name in names]

    @property
    def raw_paths(self):
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        files = self.processed_file_names
        return [osp.join(self.processed_dir, f) for f in files]

    @property
    def num_classes(self):
        return int(self.targets.max().item()) + 1 

    def download(self):
        if all([osp.exists(f) for f in self.raw_paths]):
            return
        os.makedirs(osp.expanduser(osp.normpath(self.raw_dir)))

        for name in self.raw_file_names:
            url = '{}/{}.zip'.format(self.url, name)

            filename = url.rpartition('/')[2]
            path = osp.join(self.raw_dir, filename)
            if osp.exists(path):
                print('Using exist file', filename)
            else:
                print('Downloading', url)
                data = urllib.request.urlopen(url)
                with open(path, 'wb') as f:
                    f.write(data.read())

            with zipfile.ZipFile(path, 'r') as f:
                print('Extracting', path)
                f.extractall(self.raw_dir)
            os.unlink(path)

        print('Done!')

    def process(self):
        if all([osp.exists(f) for f in self.processed_paths]):
            return

        print('Processing...')

        directory = osp.expanduser(osp.normpath(
            osp.join(self.processed_dir, self.category)))
        if not osp.exists(directory):
            os.makedirs(directory)

        idx = self.categories[self.category]
        paths = [osp.join(path, idx) for path in self.raw_paths]
        datasets = []
        for path in zip(paths[::2], paths[1::2]):
            pos_paths = sorted(glob.glob(osp.join(path[0], '*.pts')))
            y_paths = sorted(glob.glob(osp.join(path[1], '*.seg')))
            positions, ys = [], []
            for path in zip(pos_paths, y_paths):
                pos = read_txt_array(path[0])
                y = read_txt_array(path[1], dtype=torch.long)
                pos, y = fixed_points(pos, y, 2048)

                positions.append(pos.t())
                ys.append(y)

            positions = torch.stack(positions)
            ys = torch.stack(ys)
            datasets.append((positions, ys))

        train_data = torch.cat([datasets[0][0], datasets[1][0]], dim=0), torch.cat(
            [datasets[0][1], datasets[1][1]], dim=0)
        test_data = datasets[2]

        torch.save(train_data, self.processed_paths[0])
        torch.save(test_data, self.processed_paths[1])

        print('Done.')
