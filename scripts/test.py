import RoadDamageDataset
import argparse
import logs
import model_zoo
import utils
import transforms as T
import numpy as np
import torch
import sys
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint


import matplotlib.pyplot as plt
from PIL import Image


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_zoo.model_detector("fasterrcnn_resnet50", 6, device)
    model.model.load_state_dict(torch.load(sys.argv[2]))
    model.model.eval()

    dataset_test = RoadDamageDataset.RoadDamageDataset(sys.argv[1], get_transform(train=False))
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:-1974])

    print(len(dataset_test))

    imgs = [img.to(device) for img, _ in dataset_test]
    targets = [target for _, target in dataset_test]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # pick one image from the test set
    #for i in range(2):
    #img, target = dataset_test[]

    output = model.model(imgs)
    metric = MeanAveragePrecision()
    metric.update(output, targets)
    pprint(metric.compute()['map'].item())

    #print(output)