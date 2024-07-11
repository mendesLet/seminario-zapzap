import re
from collections import Counter

class DataProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def extract_classes_and_colors_from_detections(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        
        class_color_pairs = []
        
        for line in lines:
            class_match = re.search(r"Class: (\w+),", line)
            color_match = re.search(r"Color: (\w+)", line)
            
            if class_match and color_match:
                class_color_pairs.append((class_match.group(1), color_match.group(1)))
        
        return class_color_pairs

    def get_class_color_counts(self, class_color_pairs):
        return Counter(class_color_pairs)

    def save_top_class_color_pairs(self, top_class_color_pairs):
        with open(self.output_file, 'w') as f:
            for (class_name, color), count in top_class_color_pairs:
                f.write(f"Class: {class_name}, Color: {color}, Count: {count}\n")

    def process_data(self):
        class_color_pairs = self.extract_classes_and_colors_from_detections()
        class_color_counts = self.get_class_color_counts(class_color_pairs)
        
        print("Classes, cores e suas contagens:")
        for (class_name, color), count in class_color_counts.items():
            print(f"{class_name} ({color}): {count}")
        
        top_n = int(input("Digite o número de combinações classe-cor mais recorrentes que deseja salvar: "))
        top_class_color_pairs = class_color_counts.most_common(top_n)
        
        self.save_top_class_color_pairs(top_class_color_pairs)
        print(f"As {top_n} combinações classe-cor mais recorrentes foram salvas em '{self.output_file}'.")

# Exemplo de uso
if __name__ == "__main__":
    processor = DataProcessor(
        input_file='C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\detections.txt',
        output_file='top_class_color_pairs.txt'
    )
    processor.process_data()
