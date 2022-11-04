import torch
import torchvision 
import pycocotools
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from torchvision.utils import draw_bounding_boxes




if __name__ == "__main__":
    print("PyTorch Version : {}".format(torch.__version__))
    print("TorchVision Version : {}".format(torchvision.__version__))
    family = Image.open("test_images/family.jpg")
    #family.show()
    kids_play = Image.open("test_images/kids-playing.jpg")
    #kids_play.show()

    #converting images to torch tensors
    family_tensor_int = pil_to_tensor(family)
    kids_play_tensor_int = pil_to_tensor(kids_play)
    print(f"Shapes of the images: {family_tensor_int.shape}, {kids_play_tensor_int.shape}")

    #adding batch dimension
    family_tensor_int = family_tensor_int.unsqueeze(0)
    kids_play_tensor_int = kids_play_tensor_int.unsqueeze(0)
    print(f"Shapes after adding batch dimension: {family_tensor_int.shape}, {kids_play_tensor_int.shape}")

    #normalizing the images [0 - 1]
    print(f"Min value: {family_tensor_int.min()}, Max value: {family_tensor_int.max()}")
    family_tensor_float = family_tensor_int / 255.0
    kids_play_tensor_float = kids_play_tensor_int / 255.0
    print(f"Min value: {family_tensor_float.min()}, Max value: {family_tensor_float.max()}")

    #Loading the Pre-Trained PyTorch Model (Faster R-CNN with ResNet50 Backbone)
    object_detector = fasterrcnn_resnet50_fpn(weights="DEFAULT", progress=False)
    object_detector.eval() #setting the model for evaluation (don't change the weights)

    #detecting the objects in the images
    family_pred = object_detector(family_tensor_float)
    kids_play_pred = object_detector(kids_play_tensor_float)
    #print(family_pred)
    #print(kids_play_pred)

    #removing the detection if confidence is lower than 0.8
    family_pred[0]["boxes"] = family_pred[0]["boxes"][family_pred[0]["scores"] > 0.8]
    family_pred[0]["labels"] = family_pred[0]["labels"][family_pred[0]["scores"] > 0.8]
    family_pred[0]["scores"] = family_pred[0]["scores"][family_pred[0]["scores"] > 0.8]

    kids_play_pred[0]["boxes"] = kids_play_pred[0]["boxes"][kids_play_pred[0]["scores"] > 0.8]
    kids_play_pred[0]["labels"] = kids_play_pred[0]["labels"][kids_play_pred[0]["scores"] > 0.8]
    kids_play_pred[0]["scores"] = kids_play_pred[0]["scores"][kids_play_pred[0]["scores"] > 0.8]

    #loading target classes and mapping to get the correct labels of the detection
    annFile='annotations/instances_val2017.json'
    coco=COCO(annFile)
    family_labels = coco.loadCats(family_pred[0]["labels"].numpy())
    kids_play_labels = coco.loadCats(kids_play_pred[0]["labels"].numpy())
    #print(f"family labels:\n {family_labels}")
    #print(f"kids play labels:\n {kids_play_labels}")

    #visualizing bounding boxes on the original images
    family_annot_labels = ["{}-{:.2f}".format(label["name"], prob) for label, prob in zip(family_labels, family_pred[0]["scores"].detach().numpy())]

    family_output = draw_bounding_boxes(image=family_tensor_int[0],
                                        boxes=family_pred[0]["boxes"],
                                        labels=family_annot_labels,
                                        colors=["red" if label["name"]=="person" else "green" for label in family_labels],
                                        width=2
                                    )
    family_output = to_pil_image(family_output)
    family_output.show()

    kids_play_annot_labels = ["{}-{:.2f}".format(label["name"], prob) for label, prob in zip(kids_play_labels, kids_play_pred[0]["scores"].detach().numpy())]

    kids_play_output = draw_bounding_boxes(image=kids_play_tensor_int[0],
                                        boxes=kids_play_pred[0]["boxes"],
                                        labels=kids_play_annot_labels,
                                        colors=["red" if label["name"]=="person" else "green" for label in kids_play_labels],
                                        width=2,
                                        fill=True
                                    )
    kids_play_output = to_pil_image(kids_play_output)
    kids_play_output.show()






    