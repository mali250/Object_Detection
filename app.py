

import cv2

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = "labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture("Crossing Roads in Dubai (UAE) __ Strict Traffic Rules In Dubai __ Zebra Crossing.mp4")
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IndexError("Cannot open video capture.")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
   #print(ClassIndex)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (225, 0, 0), 2)
                # label = f"{classLabels[ClassInd - 1]}: {conf:.2f}"
                # cv2.putText(frame, label, (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0))
                 # Modify text color and style
                label = f"{classLabels[ClassInd - 1]}"
                cv2.putText(frame, label, (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 0, 255), thickness=2)

        cv2.imshow("video", frame)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
