import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# Caminhos dos arquivos de configuração, pesos e nomes das classes
# config_path = 'D:\\Users\\Carlo\\Documents\\VS Code\\UnoDetector\\UnoRaw.cfg'  # Substitua pelo caminho correto
config_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\UnoRaw.cfg'
# weights_path = 'D:\\Users\\Carlo\\Documents\\VS Code\\UnoDetector\\UnoRaw_best.weights'  # Substitua pelo caminho correto
weights_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\UnoRaw_best.weights'
# names_path = 'D:\\Users\\Carlo\\Documents\\VS Code\\UnoDetector\\uno.names'  # Substitua pelo caminho correto
names_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\uno.names'

# Carregar os nomes das classes
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Carregar a rede YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Configurar a rede para utilizar a CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Função para obter a saída da rede
def get_outputs(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Função para desenhar as caixas delimitadoras nas imagens
def draw_bounding_boxes(frame, outs, conf_threshold=0.8, nms_threshold=0.5):
    frame_height, frame_width = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []
    detections = []

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
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            right = left + width
            bottom = top + height
            draw_prediction(frame, class_ids[i], confidences[i], left, top, right, bottom)
            # Adicionar análise de cor aqui
            color = get_dominant_color(frame, (left, top, width, height))
            cv2.putText(frame, f'Color: {color}', (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            detections.append({
                'class_id': class_ids[i],
                'confidence': confidences[i],
                'box': (left, top, width, height),
                'color': color
            })
    return detections

def draw_prediction(frame, class_id, confidence, left, top, right, bottom):
    label = f'{classes[class_id]}: {confidence:.2f}'
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def get_dominant_color(image, box, k=4):
    # Extrair a região da bounding box
    left, top, width, height = box
    roi = image[top:top + height, left:left + width]
    
    # Converter a imagem para o espaço de cores RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Ignorar pixels brancos e pretos
    mask = np.all(roi >= [50, 50, 50], axis=-1) & np.all(roi <= [200, 200, 200], axis=-1)
    filtered_roi = roi[mask]

    if filtered_roi.size == 0:
        # Tentar ajustar a filtragem para incluir mais pixels
        mask = np.all(roi >= [30, 30, 30], axis=-1) & np.all(roi <= [220, 220, 220], axis=-1)
        filtered_roi = roi[mask]
        
        if filtered_roi.size == 0:
            # Segunda tentativa de ajuste de filtragem para incluir mais pixels
            mask = np.all(roi >= [5, 5, 5], axis=-1) & np.all(roi <= [260, 260, 260], axis=-1)
            filtered_roi = roi[mask]

            if filtered_roi.size == 0:
                # Se ainda estiver vazio, retornar uma cor padrão
                return "Unknown"

    # Usar KMeans clustering para encontrar a cor dominante
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(filtered_roi.reshape(-1, 3))
    cluster_centers = kmeans.cluster_centers_
    
    # Contar a frequência de cada cluster
    labels = kmeans.labels_
    label_counts = Counter(labels)
    dominant_cluster = label_counts.most_common(1)[0][0]
    dominant_color = cluster_centers[dominant_cluster]

    return identify_color(dominant_color)

def identify_color(dominant_color):
    r, g, b = dominant_color
    # Calcular as proporções de cada cor
    total = r + g + b
    if total == 0:
        return "Unknown"
    
    r_ratio = r / total
    g_ratio = g / total
    b_ratio = b / total
    
    # Variações de vermelho
    if r > 1.2 * g and r > 1.2 * b:
        return "Red"
    
    # Variações de verde
    if g > 1.2 * r and g > 1.2 * b:
        if g_ratio > 0.7:
            return "Green"
        elif g_ratio > 0.4:
            return "Green"
        else:
            return "Green"
    
    # Variações de azul
    if b > 1.2 * r and b > 1.2 * g:
        return "Blue"
    
    # Variações de amarelo
    if r > 1.2 * b and g > 1.2 * b:
        if r_ratio > 0.5 and g_ratio > 0.5:
            return "Yellow"
        else:
            return "Yellow"
    
    # Variações de magenta
    if r > 1.2 * g and b > 1.2 * g:
        return "MRed"
    
    # Variações de ciano
    if g > 1.2 * r and b > 1.2 * r:
        return "Green"
    
    return "Unknown"



# Abrir a câmera e realizar a detecção em tempo real
cap = cv2.VideoCapture(2)

all_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs(net))

    draw_bounding_boxes(frame, outs)
    detections = draw_bounding_boxes(frame, outs)
    all_detections.extend(detections)

    cv2.imshow('UNO Card Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Exibir as detecções salvas
for detection in all_detections:
    print(detection)