import RoadDamageDataset
import argparse
import logs
import model_zoo
import utils
import transforms as T
import numpy as np
import torch
import sys


import matplotlib.pyplot as plt
from PIL import Image





def parse_args():
    parser = argparse.ArgumentParser(description ="Road damage detection.")

    parser.add_argument("--root_dir", default="", help="Specify the path to the root of the dataset folder.")
    parser.add_argument("--train", action="store_true", 
                        help="Train the model. Otherwise the best saved model will start testing.")
    parser.add_argument("--model_name", default="fasterrcnn_resnet50", nargs="?", choices=["fasterrcnn_resnet50",
                        "fasterrcnn_resnet50v2", "fasterrcnn_mobilenetv3", "fasterrcnn_mobilenetv3_low"],
                        help="Choose a prefered pretrained model (fasterrcnn_resnet50 is default).")
    parser.add_argument("--num_epochs", default=1, help="Specify the number of epochs to train.")
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
    
    #logs.logging.info(dataset_test[2])
    #print(f"Test dataset length: {len(dataset_test)}")

    # use our dataset and defined transformations
    dataset_train = RoadDamageDataset.RoadDamageDataset(args.root_dir, get_transform(train=True))
    dataset_val = RoadDamageDataset.RoadDamageDataset(args.root_dir, get_transform(train=False))

    # split the dataset in train and validation set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-1975])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-1:])
    print(f"Train dataset length ({args.root_dir}): {len(dataset_train)}")
    print(f"Validation dataset length ({args.root_dir}): {len(dataset_val)}")

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Operation device: {device}")
    #print(torch.version.cuda)

    # num_classes which is user-defined
    num_classes = 6  # number of classes is 5 (+1 background) in our case

    # get the model using our helper function
    model = model_zoo.model_detector(args.model_name, num_classes, device)
    # move model to the right device
    #model.to_(device)
    #model.print()

    # check model's device
    print(next(model.model.parameters()).device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    # training the model
    model.train(data_loader_train, data_loader_val, args.num_epochs, optimizer, lr_scheduler)

    # pick one image from the test set
    #img, _ = dataset_test[0]

    #output = model.predict(img)
    #output.show()

    #torch.save(model.model, 'model.pt')
    #torch.save(model.model.state_dict(), 'model-parameters.pt')
    


    sys.stdout = orig_stdout
    f.close()
