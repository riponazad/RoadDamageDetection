from roboflow import Roboflow
import sys
import pprint
import torch

from torchvision import transforms
import argparse
import os

import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description ="Road damage detection Testing and producing outputs.")

    parser.add_argument("--root_dir", default="", help="Specify the path to the root folder contains test images.")
    parser.add_argument("--output_dir", default="saved_predictions", 
                        help="Specify the path to save model's prediction results.") 

    args = parser.parse_args()
    return args

#annotating labels
label_dict = {
    "D00" : 1,
    "D10" : 2,
    "D20" : 3,
    "D40" : 4
}



if __name__=='__main__':
    args = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    rf = Roboflow(api_key="GMu5JzrtMZIat8AoYbMF")
    project = rf.workspace("tdt17-bllhm").project("roaddamagedetection-zwu1t")
    model = project.version(1).model



    # infer on a local image
    #pprint.pprint(model.predict(sys.argv[1], confidence=40, overlap=30).json())

    # visualize your prediction
    #model.predict(sys.argv[1], confidence=40, overlap=30).save("prediction.jpg")

    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

    # printing prediction

    dataset_name = os.path.split(args.root_dir)

    to_tensor = transforms.ToTensor()
    open(os.path.join(args.output_dir,"roboflow"+dataset_name[1]+"_prediction.txt"), "w").close()
    
    imgs_list = os.listdir(os.path.join(args.root_dir, "test", "images"))
    j = 1

    with open(os.path.join(args.output_dir, "roboflow"+dataset_name[1]+"_prediction.txt"), "a") as f:
        for img_name in imgs_list:
            if j > 5:
                break
            i = 0
            f.write(str(img_name)+",")
            prediction = model.predict(os.path.join(args.root_dir, "test", "images", img_name),
                                        confidence=40, overlap=30).json()
            bnd_boxes = prediction['predictions']
            k = 0 # to avoid the repair class
            while i < len(bnd_boxes) and i-k < 5:
                if label_dict.get(bnd_boxes[i]['class']) == None:
                    k += 1
                    i += 1
                    continue
                xmin = round(bnd_boxes[i]['x'] - bnd_boxes[i]['width']/2)
                xmax = round(bnd_boxes[i]['x'] + bnd_boxes[i]['width']/2)
                ymin = round(bnd_boxes[i]['y'] - bnd_boxes[i]['height']/2)
                ymax = round(bnd_boxes[i]['y'] + bnd_boxes[i]['height']/2)

                f.write(str(label_dict[bnd_boxes[i]['class']])+" "+str(xmin)+" "+str(ymin)
                        +" "+str(xmax)+" "+str(ymax)+" ")
                #print(bnd_boxes[i]['class'], xmin, ymin, xmax, ymax)
                i += 1
            f.write('\n')
            j += 1

        

        #calculating bounding box

        #print(prediction['image']['height'], prediction['image']['width'])
        
        
        
        """ num_objs = len(prediction[0]['labels'])
        labels = prediction[0]['labels'].cpu()
        b_boxs = prediction[0]['boxes'].cpu()
        #print(labels)
        #print(b_boxs)
        with open(os.path.join(args.output_dir,args.model_name+dataset_name[1]+"_prediction.txt"), "a") as f:
            i = 0
            f.write(str(img_name)+",")
            print(img_name+",", end='')
            while i < num_objs and i < 5:
                f.write(str(labels[i].item())+" "+str(round(b_boxs[i][0].item()))+" "+str(round(b_boxs[i][1].item()))
                        +" "+str(round(b_boxs[i][2].item()))+" "+str(round(b_boxs[i][3].item()))+" ")
                print(str(labels[i].item())+" "+str(round(b_boxs[i][0].item()))+" "+str(round(b_boxs[i][1].item()))
                        +" "+str(round(b_boxs[i][2].item()))+" "+str(round(b_boxs[i][3].item()))+" ", end='')
                i += 1
            f.write('\n')
            print() """

    #print(output)