"""
训练完后运行此脚本，按 size 拆 mAP：
  python eval_per_size.py --weights runs/detect/train/weights/best.pt
"""
import os, argparse
from ultralytics import YOLO

BASE = r'E:\桌面\模块缝合\the visual computer（模板拒）\dataset\CCTSDB'

SIZES = ['XS', 'S', 'M', 'L', 'XL', 'Mixed']


def eval_size(model, size):
    """验证单个 size 子集，直接指定数据路径，不需要额外 yaml"""
    data = {
        'path': BASE,
        'val':  f'images/size_test/{size}',
        'names': {0: 'mandatory', 1: 'prohibitory', 2: 'warning'},
    }
    result = model.val(data=data, split='val', verbose=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='trained model .pt path')
    args = parser.parse_args()

    model = YOLO(args.weights)

    # 1. 完整 val
    print('Evaluating full val (1500 images)...')
    full = model.val(
        data='CCTSDB_data.yaml',
        split='val',
        verbose=False,
    )

    # 2. 逐个 size 子集
    print('\n' + '=' * 55)
    print(f'{"Size":<8} {"Images":<8} {"mAP@50":<10} {"mAP@50:95":<10}')
    print('=' * 55)

    for size in SIZES:
        r = eval_size(model, size)
        imgs = r.seen
        m50  = r.box.map50
        m5095 = r.box.map
        print(f'{size:<8} {imgs:<8} {m50:<10.4f} {m5095:<10.4f}')

    print('=' * 55)
    print(f'{"Overall":<8} {full.seen:<8} {full.box.map50:<10.4f} {full.box.map:<10.4f}')


if __name__ == '__main__':
    main()
