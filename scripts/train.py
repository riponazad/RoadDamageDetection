import RoadDamageDataset
import argparse
import model_zoo
import utils
import transforms as T
import numpy as np
import torch
import sys
import os


import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image




def parse_args():
    parser = argparse.ArgumentParser(description ="Road damage detection.")

    parser.add_argument("--root_dir", default="", help="Specify the path to the root of the dataset folder.")
    parser.add_argument("--train", action="store_true", 
                        help="Train the model. Otherwise the best saved model will start testing.")
    parser.add_argument("--model_name", default="fasterrcnn_resnet50", nargs="?", choices=["fasterrcnn_resnet50",
                        "fasterrcnn_resnet50v2", "fasterrcnn_mobilenetv3", "fasterrcnn_mobilenetv3_low",
                        "fcos_resnet50_fpn"],
                        help="Choose a prefered pretrained model (fasterrcnn_resnet50 is default).")
    parser.add_argument("--num_epochs", default=1, type=int, help="Specify the number of epochs to train.")
    parser.add_argument("--model_dir", default="saved_model", help="Specify the path to save model.") 

    args = parser.parse_args()
    return args


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
    args = parse_args()
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    #sys.stdout = f

    #dataset_test = RoadDamageDataset.RoadDamageDatasetTest(args.root_dir, get_transform(train=False))
    

    # use our dataset and defined transformations
    dataset_train = RoadDamageDataset.RoadDamageDataset(args.root_dir, get_transform(train=True))
    dataset_val = RoadDamageDataset.RoadDamageDataset(args.root_dir, get_transform(train=False))

    # split the dataset in train and validation set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    ln = len(indices)
    n = int(ln*0.75)
    #print(ln, n)
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-(ln-n)])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-(ln-n):])
    print(f"Train dataset length: {len(dataset_train)}")
    print(f"Validation dataset length: {len(dataset_val)}")

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=6, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=2, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Operation device: {device}")
    #print(torch.version.cuda)

    # num_classes which is user-defined
    num_classes = 6  # number of classes is 4 (+1 background) in our case


    # get the model using our helper function
    #model = model_zoo.model_detector(args.model_name, num_classes, device)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # move model to the right device
    model.to(device)
    tmp_model = args.model_name + ".pt"
    model.load_state_dict(torch.load(os.path.join(args.model_dir,tmp_model)))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)


    for epoch in range(args.num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)


    # look for a saved model (weights) to retrieve
    """ tmp_model = args.model_name + ".pt"
    saved_models = os.listdir(args.model_dir)
    for mdl in saved_models:
        if str(mdl) == tmp_model:
            print("Previous trained model is retrieved.")
            model.model.load_state_dict(torch.load(os.path.join(args.model_dir,tmp_model))) """

    # move model to the right device
    #model.to_(device)
    #model.print()

    # check model's device
    #print(next(model.model.parameters()).device)

    # construct an optimizer
    #params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005,
     #                           momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
      #                                          step_size=3,
       #                                         gamma=0.1)

    # training the model
    #model.train(data_loader_train, data_loader_val, args.num_epochs, optimizer, lr_scheduler)

    # pick one image from the test set
    #img, _ = dataset_test[0]

    #output = model.predict(img)
    #output.show()

    #torch.save(model.model, 'model.pt')
    #torch.save(model.model.state_dict(), 'model-parameters.pt')
    


    sys.stdout = orig_stdout
    f.close()
