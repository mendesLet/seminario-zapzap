import cv2
import time
import collections

cards_dict = {
    '10C': '10 of Clubs', '10D': '10 of Diamonds', '10H': '10 of Hearts', '10S': '10 of Spades',
    '2C': '2 of Clubs', '2D': '2 of Diamonds', '2H': '2 of Hearts', '2S': '2 of Spades',
    '3C': '3 of Clubs', '3D': '3 of Diamonds', '3H': '3 of Hearts', '3S': '3 of Spades',
    '4C': '4 of Clubs', '4D': '4 of Diamonds', '4H': '4 of Hearts', '4S': '4 of Spades',
    '5C': '5 of Clubs', '5D': '5 of Diamonds', '5H': '5 of Hearts', '5S': '5 of Spades',
    '6C': '6 of Clubs', '6D': '6 of Diamonds', '6H': '6 of Hearts', '6S': '6 of Spades',
    '7C': '7 of Clubs', '7D': '7 of Diamonds', '7H': '7 of Hearts', '7S': '7 of Spades',
    '8C': '8 of Clubs', '8D': '8 of Diamonds', '8H': '8 of Hearts', '8S': '8 of Spades',
    '9C': '9 of Clubs', '9D': '9 of Diamonds', '9H': '9 of Hearts', '9S': '9 of Spades',
    'AC': 'Ace of Clubs', 'AD': 'Ace of Diamonds', 'AH': 'Ace of Hearts', 'AS': 'Ace of Spades',
    'JC': 'Jack of Clubs', 'JD': 'Jack of Diamonds', 'JH': 'Jack of Hearts', 'JS': 'Jack of Spades',
    'KC': 'King of Clubs', 'KD': 'King of Diamonds', 'KH': 'King of Hearts', 'KS': 'King of Spades',
    'QC': 'Queen of Clubs', 'QD': 'Queen of Diamonds', 'QH': 'Queen of Hearts', 'QS': 'Queen of Spades'
}

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
# current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Configure path to project
default_path = "cardDetector"

cfg_path = "PlayingCardsRaw.cfg"
print(cfg_path)
weights_path = "PlayingCardsRaw_best.weights"

class_names = []
# with open("coco.names", "r") as f:
with open("cards.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# 0 is for webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./imgAndVideo/video.mp4")

net = cv2.dnn.readNet(cfg_path, weights_path)

# Input model size acording to cfg file
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255) #swapRB=True
# model.setInputParams(size=(512, 288), scale=1/255) #swapRB=True
# model.setInputParams(size=(608, 608), scale=1/255) #swapRB=True

frame_history = collections.deque(maxlen=40)
score_history = collections.deque(maxlen=40)
class_counts = collections.defaultdict(int)
score_counts = collections.defaultdict(int)
current_class = None

# Frame reading
while True:
    _, frame = cap.read()
    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end = time.time()

    detected_class = None
    detected_score = 0
    if len(classes) > 0 and scores[0] > 0.1:
        detected_class = class_names[classes[0]]
        detected_score = scores[0]
    
    frame_history.append(detected_class)
    score_history.append(detected_score)
    
    class_counts.clear()
    score_counts.clear()    
    for c, s in zip(frame_history, score_history):
        if c is not None:
            class_counts[c] += 1
            if s > 0.3:
                score_counts[c] += 1

    for classid, count in class_counts.items():
        if count > 30 and score_counts[classid] > 25:
            if current_class != classid:
                print(cards_dict[classid])
                current_class = classid
            break

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid]} : {score}"

        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("detections", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()