from roboflow import Roboflow
import sys
import pprint

rf = Roboflow(api_key="GMu5JzrtMZIat8AoYbMF")
project = rf.workspace("tdt17-bllhm").project("roaddamagedetection-zwu1t")
model = project.version(1).model

# infer on a local image
pprint.pprint(model.predict(sys.argv[1], confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())