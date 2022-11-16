import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

#annotating labels back
label_dict_ = {
    1 : "D00",
    2 : "D10",
    3 : "D20",
    4 : "D40",
    5 : "Repair"
}

#colors
label_colors = {
    "D00" : 'red',
    "D10" : 'green',
    "D20" : 'orange',
    "D40" : 'blue',
    "Repair" : 'purple'
}

class model_detector():
    def __init__(self, model_name, num_classes, device=torch.device('cpu')):

        self.model_name = model_name
        self.device = device
        # to save the best validation epoch's model
        self.best_map = -1.0

        if self.model_name=="fasterrcnn_resnet50":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        elif self.model_name=="fasterrcnn_resnet50v2":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        elif self.model_name=="fasterrcnn_mobilenetv3":
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        elif self.model_name=="fasterrcnn_mobilenetv3_low":
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.model.to(self.device)

    def print(self):
        print(self.model)

    def parameters(self):
        return self.model.parameters()

    def train(self, data_loader_train, data_loader_val, num_epochs, optimizer, lr_scheduler):
        #For saving the best model
        dataset_val = data_loader_val.dataset
        imgs = [img.to(self.device) for img, _ in dataset_val]
        targets = [target for _, target in dataset_val]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        metric = MeanAveragePrecision()

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, data_loader_train, self.device, epoch, print_freq=10)
            #pred = self.model(data_loader_train.dataset)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, data_loader_val, device=self.device) 
            
            output = self.model(imgs)
            metric.update(output, targets)
            mAP = metric.compute()['map'].item()
            print(f"Dataset length: {len(dataset_val)}\n mAP:{mAP}")
            if mAP > self.best_map:
                torch.save(self.model.state_dict(), "saved_model/" + self.model_name +
                                                         "-best-epoch"+ str(epoch) +".pt")
                self.best_map = mAP

        torch.save(self.model.state_dict(), "saved_model/" + self.model_name +
                                                         "-last-epoch"+ str(epoch) +".pt")



    def predict(self, img):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([img.to(self.device)])

        prediction[0]["boxes"] = prediction[0]["boxes"][prediction[0]["scores"] > 0.8]
        prediction[0]["labels"] = prediction[0]["labels"][prediction[0]["scores"] > 0.8]
        prediction[0]["scores"] = prediction[0]["scores"][prediction[0]["scores"] > 0.8]

        #prediction = prediction.to('cpu')
        prediction_labels = [label_dict_[i] for i in prediction[0]["labels"].cpu().numpy()]
        sc = prediction[0]["scores"].cpu().detach().numpy()

        prediction_annot_labels = ["{}-{:.2f}".format(label, prob) for label, prob in zip(prediction_labels, sc )]

        output = draw_bounding_boxes(image=torch.tensor(img*255, dtype=torch.uint8),
                                    boxes=prediction[0]["boxes"],
                                    labels=prediction_annot_labels,
                                    colors=[label_colors[label] for label in prediction_labels],
                                    width=2
                                    )
        output = to_pil_image(output)
        return output 
