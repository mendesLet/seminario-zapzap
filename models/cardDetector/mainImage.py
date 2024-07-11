import cv2
import os
import datetime
import argparse

parser = argparse.ArgumentParser(description="Inference on an image")
parser.add_argument('--path', type=str, default='./plateRecognition',
                    help='Path to model files')
parser.add_argument('--imgName', type=str, default='img3.jpg',
                    help='Name of image to be used, with the file extension')
parser.add_argument('--saveInference', type=bool, default=False,
                help=f'Choose if you want to save your inference')
args = parser.parse_args()

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Configure path to project
default_path = args.path
if default_path ==  "./cardDetector":
    cfg_path = default_path + "/PlayingCardsRaw.cfg"
    weights_path = default_path + "/PlayingCardsRaw_best.weights"

else:
    print("Not valid path, terminating...")
    exit()

class_names = []
with open("./cardDetector/cards.names", "r") as f:  
    class_names = [cname.strip() for cname in f.readlines()]

# Put image path here
image_path = f"./imgAndVideo/{args.imgName}"
frame = cv2.imread(image_path)
net = cv2.dnn.readNet(cfg_path, weights_path)
model = cv2.dnn_DetectionModel(net)

# Configure network dimensions according to cfg file
model.setInputParams(size=(608, 608), scale=1/255) 
classes, scores, boxes = model.detect(frame, 0.1, 0.2)

for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = f"{class_names[classid]} : {score:.2f}"
    cv2.rectangle(frame, box, color, 2)
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

inferenceDir = "./inference"

if args.saveInference:
    filename = os.path.join(inferenceDir, current_time + ".jpg")
    cv2.imwrite(filename, frame)

frame = cv2.resize(frame, (1920, 1080))

# Hit "esc" to leave window, or bugs will occur
cv2.imshow("Detections", frame)
cv2.waitKey(0) 
cv2.destroyAllWindows()
