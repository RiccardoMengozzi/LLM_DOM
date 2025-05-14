# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# model = YOLO("model/best.pt")
# image = plt.imread("images/0.png")
# results = model(image[0])
# plt.imshow(results[0])
# plt.show()

import cv2
from ultralytics import YOLO
import supervision as sv


image_path = "dog.jpg"
image = cv2.imread(image_path)

model = YOLO("model/best.pt")

results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = image.copy()
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)