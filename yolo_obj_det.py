import cv2, pandas
import numpy as np


net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

'''
print(len(classes))
print(classes)
'''

'''
if you want to use webcam then : video = cv2.VideoCapture(0)
if you have video at any other location give the path in the round braces
'''
video = cv2.VideoCapture("fruits.mp4")

while True:
    _, img = video.read()
    height, width, _  = img.shape


    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), True, crop = False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layersOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []


    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)


                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:

            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+30), font, 2, color, 2)

    cv2.imshow('Image', img )
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


