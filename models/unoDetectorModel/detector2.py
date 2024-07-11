import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import time
import re

class UnoCardDetector:
    def __init__(self, config_path, weights_path, names_path):
        self.config_path = config_path
        self.weights_path = weights_path
        self.names_path = names_path
        self.classes = self.load_classes()
        self.net = self.load_network()
        self.stop = False
        self.all_detections = []

    def load_classes(self):
        with open(self.names_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def load_network(self):
        net = cv2.dnn.readNet(self.weights_path, self.config_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def get_outputs(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_boxes(self, frame, outs, conf_threshold=0.8, nms_threshold=0.5):
        frame_height, frame_width = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                right = left + width
                bottom = top + height
                self.draw_prediction(frame, class_ids[i], confidences[i], left, top, right, bottom)
                color = self.get_dominant_color(frame, (left, top, width, height))
                cv2.putText(frame, f'Color: {color}', (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                detections.append({
                    'class_id': class_ids[i],
                    'confidence': confidences[i],
                    'box': (left, top, width, height),
                    'color': color
                })
        return detections

    def draw_prediction(self, frame, class_id, confidence, left, top, right, bottom):
        label = f'{self.classes[class_id]}: {confidence:.2f}'
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def get_dominant_color(self, image, box, k=4):
        left, top, width, height = box
        roi = image[top:top + height, left:left + width]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mask = np.all(roi >= [50, 50, 50], axis=-1) & np.all(roi <= [200, 200, 200], axis=-1)
        filtered_roi = roi[mask]

        if filtered_roi.size == 0:
            mask = np.all(roi >= [30, 30, 30], axis=-1) & np.all(roi <= [220, 220, 220], axis=-1)
            filtered_roi = roi[mask]
            if filtered_roi.size == 0:
                mask = np.all(roi >= [5, 5, 5], axis=-1) & np.all(roi <= [260, 260, 260], axis=-1)
                filtered_roi = roi[mask]
                if filtered_roi.size == 0:
                    return "Unknown"

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(filtered_roi.reshape(-1, 3))
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_counts = Counter(labels)
        dominant_cluster = label_counts.most_common(1)[0][0]
        dominant_color = cluster_centers[dominant_cluster]

        return self.identify_color(dominant_color)

    def identify_color(self, dominant_color):
        r, g, b = dominant_color
        total = r + g + b
        if total == 0:
            return "Unknown"

        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total

        if r > 1.2 * g and r > 1.2 * b:
            return "Red"
        if g > 1.2 * r and g > 1.2 * b:
            if g_ratio > 0.7:
                return "Green"
            elif g_ratio > 0.4:
                return "Green"
            else:
                return "Green"
        if b > 1.2 * r and b > 1.2 * g:
            return "Blue"
        if r > 1.2 * b and g > 1.2 * b:
            if r_ratio > 0.5 and g_ratio > 0.5:
                return "Yellow"
            else:
                return "Yellow"
        if r > 1.2 * g and b > 1.2 * g:
            return "MRed"
        if g > 1.2 * r and b > 1.2 * r:
            return "Green"

        return "Unknown"

    def stop_capture(self):
        self.stop = True

    def run(self, camera_index=1, duration=30):
        with open('detections.txt', 'w') as f:
            pass
        
        self.stop = False
        cap = cv2.VideoCapture(camera_index)
        start_time = time.time()

        while cap.isOpened() and (time.time() - start_time < duration) and not self.stop:
            ret, frame = cap.read()
            if not ret:
                break

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.get_outputs())

            detections = self.draw_bounding_boxes(frame, outs)
            self.all_detections.extend(detections)
            print("Detecções no frame atual:", detections)

            cv2.imshow('UNO Card Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop = True
                break

        cap.release()
        cv2.destroyAllWindows()

        print("Todas as detecções:")
        for detection in self.all_detections:
            print(detection)

        self.save_detections()

    def save_detections(self):
        if not self.all_detections:
            print("Nenhuma detecção para salvar.")
            return

        with open('detections.txt', 'w') as f:
            for detection in self.all_detections:
                class_name = self.classes[detection['class_id']]
                f.write(f"Class: {class_name}, Confidence: {detection['confidence']:.2f}, Box: {detection['box']}, Color: {detection['color']}\n")

        print("Detecções salvas em 'detections.txt'.")

def main():
    config_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\UnoRaw.cfg'
    weights_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\UnoRaw_best.weights'
    names_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\uno.names'

    detector = UnoCardDetector(config_path, weights_path, names_path)
    with open('detections.txt', 'w') as f:
        pass
    detector.run(duration=5)  # Executa a detecção por 5 segundo

if __name__ == "__main__":
    main()
