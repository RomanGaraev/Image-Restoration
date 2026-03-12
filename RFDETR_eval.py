import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRLarge()

image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
detections = model.predict(image, threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
annotated_image.show()