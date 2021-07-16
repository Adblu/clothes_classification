import numpy as np
from mmcv import Config
import os.path as osp

from src.train_config import CLASS_COUNT, DATA_PREFIX, TRAIN_ANNOTS, CLASSES, TEST_ANNOTS, WORK_DIR, \
    IMAGES_LABELS, EPOCH_COUNT, TRANSFER_LEARNING, CHECKPOINT_PATH
from mmcls.apis import set_random_seed
import mmcv

from mmcls.datasets import build_dataset
from mmcls.models import build_classifier

from mmcls.datasets import DATASETS, BaseDataset
from glob import glob
import json


@DATASETS.register_module()
class ClothesDataset(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos


def bring_dataset():
    def parse_images_data(phase):
        images = glob(IMAGES_LABELS + phase + '/*/*.*')

        classes = {'dress': '1',
                   'hat': '2',
                   'longsleeve': '3',
                   'outwear': '4',
                   'pants': '5',
                   'shirt': '6',
                   'shoes': '7',
                   'shorts': '8',
                   'skirt': '9',
                   't-shirt': '10'
                   }

        _imgs_list = []
        _cls_list = []

        with open('/home/n/Documents/STER/src/data/txt/' + phase + '.txt', 'wt') as f:
            for image in images:
                file = image.split('/')[-1]
                _class = classes[image.split('/')[-2]]
                _imgs_list.append(file)
                _cls_list.append(_class)

                f.write(file + ' ' + _class + '\n')

        return np.array(_imgs_list), np.array(_cls_list)

    X_train, X_test = parse_images_data('train')
    y_train, y_test = parse_images_data('test')
    return X_train, X_test, y_train, y_test


def prepare_config(CONFIG_PATH):
    cfg = Config.fromfile(CONFIG_PATH)

    cfg.dataset_type = 'ClothesDataset'
    cfg.data.train.type = cfg.dataset_type
    cfg.data.val.type = cfg.dataset_type
    cfg.data.test.type = cfg.dataset_type

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu = 1

    cfg.img_norm_cfg = dict(
        mean=[124.508, 116.050, 106.438], std=[58.577, 57.310, 57.437], to_rgb=True)

    cfg.data.train.data_prefix = DATA_PREFIX
    cfg.data.train.ann_file = TRAIN_ANNOTS
    cfg.data.train.classes = CLASSES

    cfg.data.val.data_prefix = DATA_PREFIX
    cfg.data.val.ann_file = TEST_ANNOTS
    cfg.data.val.classes = CLASSES

    cfg.data.test.data_prefix = DATA_PREFIX
    cfg.data.test.ann_file = TEST_ANNOTS
    cfg.data.test.classes = CLASSES
    cfg.evaluation['metric_options'] = {'topk': (1)}

    cfg.model.head.num_classes = CLASS_COUNT

    cfg.model.head.topk = (1)

    cfg.optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)

    cfg.lr_config = dict(policy='step', step=[1])
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=EPOCH_COUNT)

    if TRANSFER_LEARNING:
        cfg.load_from = CHECKPOINT_PATH

    cfg.work_dir = WORK_DIR

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    cfg.dump(WORK_DIR + '/model_config.py')

    return cfg


def prepare_model(cfg):
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    model = build_classifier(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    return model, datasets


def bring_stats(file):

    validation = []
    accuracy_list = []

    with open(file, 'r') as f:
        contents = f.readlines()

    for i in range(1, len(contents)):
        tmp = json.loads(contents[i].replace('\n', ''))

        if 'val' in tmp['mode']:
            validation.append(tmp)

    for j in range(len(validation)):
        accuracy_list.append(validation[j]['accuracy'])

    return accuracy_list
