import argparse
import numpy as np
import json
import os

DATA_DIR = 'data/coco/annotations'

def prepare_coco_data(seed=1, percent=30.0, version=2017):
  """Prepare Cityscapes data for Semi-supervised learning

  Args:
    seed: random seed for data split
    percent: percentage of labeled data
  """
  def _save_anno(name, images, annotations):
    """Save annotation
    """
    print('>> Processing data {}.json saved ({} images {} annotations)'.format(
        name, len(images), len(annotations)))
    new_anno = {}
    new_anno['images'] = images
    new_anno['annotations'] = annotations
    new_anno['categories'] = anno['categories']

    with open(
        '{root}/{save_name}.json'.format(
            save_name=name, root=DATA_DIR),
        'w') as f:
      json.dump(new_anno, f)
    print('>> Data {}.json saved ({} images {} annotations)'.format(
        name, len(images), len(annotations)))

  np.random.seed(seed)
  
  anno = json.load(open(os.path.join(DATA_DIR, 'instances_train2017.json')))

  image_list = anno['images']
  labeled_tot = int(percent / 100. * len(image_list))
  #labeled_ind = np.random.choice(range(len(image_list)), size=labeled_tot)
  labeled_ind = np.arange(len(image_list))
  np.random.shuffle(labeled_ind)
  labeled_ind = labeled_ind[0:labeled_tot]

  labeled_id = []
  labeled_images = []
  unlabeled_images = []
  labeled_ind = set(labeled_ind)
  for i in range(len(image_list)):
    if i in labeled_ind:
      labeled_images.append(image_list[i])
      labeled_id.append(image_list[i]['id'])
    else:
      unlabeled_images.append(image_list[i])

  # get all annotations of labeled images
  labeled_id = set(labeled_id)
  labeled_annotations = []
  unlabeled_annotations = []
  for an in anno['annotations']:
    if an['image_id'] in labeled_id:
      labeled_annotations.append(an)
    else:
      unlabeled_annotations.append(an)

  # save labeled and unlabeled
  save_name = 'instances_train2017.{seed}@{tot}'.format(
      seed=seed, tot=int(percent))
  _save_anno(save_name, labeled_images, labeled_annotations)
  save_name = 'instances_train2017.{seed}@{tot}-unlabeled'.format(
      seed=seed, tot=int(percent))
  _save_anno(save_name, unlabeled_images, unlabeled_annotations)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--percent', type=float, default=30)
  parser.add_argument('--seed', type=int, help='seed', default=1)

  args = parser.parse_args()
  prepare_coco_data(args.seed, args.percent)
