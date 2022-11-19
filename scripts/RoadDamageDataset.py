import os
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET

#annotating labels
label_dict = {
    "D00" : 1,
    "D10" : 2,
    "D20" : 3,
    "D40" : 4
}


def isValid(root, annot):
    bbox_path = os.path.join(root, "train", "annotations", "xmls", annot)
    tree = ET.parse(bbox_path)
    #print(bbox_path)
    root = tree.getroot()
    if root.find('object') == None:
        return False
    
    for obj in root.iter('object'):
        if obj.find('bndbox') == None:
            return False
        elif label_dict.get(obj.find('name').text) == None:
            return False
        
    return True


class RoadDamageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        img_files = list(sorted(os.listdir(os.path.join(root, "train", "images"))))
        bbox_files = list(sorted(os.listdir(os.path.join(root, "train", "annotations", "xmls"))))
        self.imgs = [] 
        self.bboxes = []
        # load all image files, sorting them to
        # ensure that they are aligned
        for idx in range(len(bbox_files)):
            if isValid(root, bbox_files[idx]):
                self.imgs.append(img_files[idx])
                self.bboxes.append(bbox_files[idx])


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "train", "images", self.imgs[idx])
        bbox_path = os.path.join(self.root, "train", "annotations", "xmls", self.bboxes[idx])
        img = Image.open(img_path).convert("RGB")

        #reading bounding box information from xml file
        tree = ET.parse(bbox_path)
        #print(bbox_path)
        root = tree.getroot()
        #print(root.tag, root.attrib)

        boxes = []
        labels = []
        for obj in root.iter('object'):
          #print(obj[0].text)
          bounding_box = obj[4]
          xmin = float(bounding_box[0].text)
          xmax = float(bounding_box[2].text)
          ymin = float(bounding_box[1].text)
          ymax = float(bounding_box[3].text)
          boxes.append([xmin, ymin, xmax, ymax])
          labels.append(label_dict[obj[0].text])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        #print(boxes.shape)
        area = torch.tensor([])
        if len(labels) > 0:
          area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



class RoadDamageDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "test", "images"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "test", "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        image_id = torch.tensor([idx])
        target = {}
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)