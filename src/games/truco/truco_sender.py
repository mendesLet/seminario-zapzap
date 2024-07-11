import cv2
import time
import numpy as np
from collections import deque, Counter
import requests
import argparse

parser = argparse.ArgumentParser(description='Truco Sender')
parser.add_argument('--default_path', type=str, default='../models/greyCardDetector/', help='Path to the configuration file')
args = parser.parse_args()

default_path = args.default_path

def apply_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red_mask = (image[:,:,0] > image[:,:,1]) & (image[:,:,0] > image[:,:,2])
    image[red_mask, 0] = (image[red_mask, 0] * 0.6).astype(np.uint8)
    gray_image = 0.21 * image[:,:,0] + 0.72 * image[:,:,1] + 0.07 * image[:,:,2]
    gray_image = gray_image.astype(np.uint8)
    gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return gray_image_bgr

def send_prediction(prediction):
    url = 'http://localhost:5000/receive_prediction'
    data = {'prediction': prediction}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print('Prediction sent successfully')
        else:
            print('Failed to send prediction')
    except Exception as e:
        print(f'Error sending prediction: {e}')

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

cfg_path = default_path + "/PlayingCardsProprietaryLuminousGrey.cfg"
weights_path = default_path + "/PlayingCardsProprietaryLuminousGrey_best.weights"

class_names = []
with open(default_path + "/cards.names", "r") as f:  
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture(0)
net = cv2.dnn.readNet(cfg_path, weights_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)

predictions = deque()
buffer_time = 7
cycle_start_time = time.time()
last_prediction = None

first_inference = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = apply_filter(frame)

    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end = time.time()

    current_time = time.time()
    predictions.append((current_time, classes))

    while predictions and current_time - predictions[0][0] > buffer_time:
        predictions.popleft()

    all_predictions = [class_names[classid] for _, classes in predictions for classid in classes]
    top_predictions = Counter(all_predictions).most_common(3)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid]} : {score:.2f}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cycle_time = round(buffer_time - (current_time - cycle_start_time), 1)
    if cycle_time <= 0:
        if first_inference:
            top_predictions_list = [pred[0] for pred in top_predictions]
            if top_predictions_list:
                last_prediction = top_predictions_list
            first_inference = False
        else:
            most_common_class = Counter(all_predictions).most_common(1)
            if most_common_class:
                last_prediction = [most_common_class[0][0]]
        
        if last_prediction:
            print("Top Predicted Cards:", last_prediction)
            send_prediction(last_prediction)
        
        cycle_start_time = current_time

    timer_label = f"Time: {cycle_time}s"
    cv2.putText(frame, timer_label, (frame.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, timer_label, (frame.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("detections", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
