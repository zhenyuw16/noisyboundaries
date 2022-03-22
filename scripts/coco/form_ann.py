import json
from copy import deepcopy
import numpy as np
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='forming annotation file')
    parser.add_argument('plname')
    parser.add_argument('annname')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    fann = args.annname
    fpl = args.plname
    
    dett = json.load(open(fpl.split('.')[0] + '_t.segm.json'))
    
    a = json.load(open('data/coco/annotations/instances_train2017.1@10-unlabeled.json'))
    aaa = json.load(open('data/coco/annotations/instances_train2017.1@10.json'))
    
    iiid = [i['id'] for i in aaa['annotations']]
    print(len(a['annotations']))
    
    b = dett

    j = max(iiid)
    for i in range(len(b)):
        x1, x2, y1, y2 = [b[i]['bbox'][0], b[i]['bbox'][0]+b[i]['bbox'][2], b[i]['bbox'][1], b[i]['bbox'][1]+b[i]['bbox'][3]]
        b[i]['area'] = b[i]['bbox'][2] * b[i]['bbox'][3]
        j = j + 1
        b[i]['id'] = j
    
    a['annotations'] = b
    
    json.dump(a, open(fann,'w'))
