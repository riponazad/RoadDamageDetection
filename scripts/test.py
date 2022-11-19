import RoadDamageDataset
import argparse
import model_zoo
from torchvision import transforms
import torch
import os

import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description ="Road damage detection Testing and producing outputs.")

    parser.add_argument("--root_dir", default="", help="Specify the path to the root folder contains test images.")
    parser.add_argument("--model_name", default="fasterrcnn_resnet50", nargs="?", choices=["fasterrcnn_resnet50",
                        "fasterrcnn_resnet50v2", "fasterrcnn_mobilenetv3", "fasterrcnn_mobilenetv3_low",
                        "fcos_resnet50_fpn"],
                        help="Choose a prefered pretrained model (fasterrcnn_resnet50 is default).")
    parser.add_argument("--output_dir", default="saved_predictions", 
                        help="Specify the path to save model's prediction results.") 

    args = parser.parse_args()
    return args


""" def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms) """


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_zoo.model_detector(args.model_name, 6, device)
    print("Pretrained model is loaded.")
    tmp_model = args.model_name + ".pt"
    model.model.load_state_dict(torch.load(os.path.join("saved_model",tmp_model)))
    #model.model.eval()

    #dataset_test = RoadDamageDataset.RoadDamageDatasetTest(args.root_dir, get_transform(train=False))

    to_tensor = transforms.ToTensor()
    
    imgs_list = os.listdir(os.path.join(args.root_dir, "test", "images"))
    j = 1
    for img_name in imgs_list:
        if j > 5:
            break
        img = Image.open(os.path.join(args.root_dir, "test", "images", img_name))
        prediction = model.predict(to_tensor(img))
        num_objs = len(prediction[0]['labels'])
        labels = prediction[0]['labels'].cpu()
        b_boxs = prediction[0]['boxes'].cpu()
        print(labels)
        print(b_boxs)
        with open(os.path.join(args.output_dir,args.model_name+"_prediction.txt"), "w") as f:
            i = 0
            f.write(str(img_name)+",")
            while i < num_objs and i < 5:
                f.write(str(labels[i]+" "+b_boxs[i][0]+" "+b_boxs[i][1]+" "+b_boxs[i][2]+" "+b_boxs[i][3]+"\n"))
                i += 1
        j += 1

    #print(output)