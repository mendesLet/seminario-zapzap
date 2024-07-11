import os
import tkinter as tk
from tkinter import Button
from dotenv import load_dotenv
import google.generativeai as genai
import time
from tkinter import Tk, Button
from detector2 import UnoCardDetector
from preprocess import DataProcessor

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave da API da variável de ambiente
gemini_api_key = os.getenv("GEMINI_API_KEY")
print(f"Chave da API: {gemini_api_key}")

# Configurar a chave da API
genai.configure(api_key=gemini_api_key)

# Configuração do modelo
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

print("Modelo configurado com sucesso.")


class UnoGame:
    def __init__(self, detector, data_processor):
        self.player_hand = []
        self.discard_pile = []
        self.detector = detector
        self.data_processor = data_processor  # Instância de DataProcessor
        print("Jogo UNO iniciado. Inicializando a mão do jogador...")
        self.initialize_player_hand()

    def initialize_player_hand(self):
        input("Pressione Enter para inicializar a mão do jogador...")
        self.update_hand()

    def update_hand(self):
        input("Pressione Enter para atualizar a mão do jogador...")
        print("Atualizando a mão do jogador...")
        with open('detections.txt', 'w') as f:
            pass
        self.detector.run(duration=5)
        self.data_processor.process_data()  # Processa os dados depois da detecção
        self.player_hand = self.read_class_color_pairs('top_class_color_pairs.txt')
        print(f"Mão atualizada: {self.player_hand}")

    def update_discard_pile(self):
        input("Pressione Enter para atualizar a pilha de descarte...")
        print("Atualizando a pilha de descarte...")
        with open('detections.txt', 'w') as f:
            pass
        self.detector.run(duration=5)
        self.data_processor.process_data()  # Processa os dados depois da detecção
        new_top_card = self.read_class_color_pairs('top_class_color_pairs.txt')
        if new_top_card:
            self.discard_pile.append(new_top_card)
        print(f"Pilha de descarte atualizada: {self.discard_pile}")

    def read_class_color_pairs(self, filename):
        print(f"Lendo arquivo: {filename}")
        with open(filename, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def player_turn(self):
        print("Iniciando turno do jogador...")
        self.update_hand()
        self.update_discard_pile()

    def assistant_turn(self):
        print("Iniciando turno do assistente...")
        decision = self.uno_assistant()
        print(f"Decisão do assistente: {decision}")


    def uno_assistant(self):
        if self.discard_pile:
            top_card = self.discard_pile[-1]
        else:
            top_card = 'Nenhuma carta'
        prompt = f"Você é um dos jogadores de uma partida de UNO, está jogando contra outro jogador.Sua mão: {self.player_hand}, Topo da pilha de descarte: {self.discard_pile}. Usando a carta da pilha de descarte, construa a melhor estratégia para eliminar o mais rápido possível todas suas cartas, respeitando as regras oficiais do UNO.A saída deve ser no seguinte formato: ##Exemplos## Output: Jogue as cartas 7 verde, 7 vermelho e 7 azul juntas. Output: Jogue a Comprar 2 Verde."
        response = model.generate_content(prompt)  # Ajuste para o nome correto do método
        print(f"Prompt enviado ao modelo: {prompt}")
        return response.text

    def play_game(self):
        print("Iniciando o jogo...")
        while self.player_hand:
            self.player_turn()
            self.assistant_turn()
        print("Jogo terminado - não há mais cartas na mão do jogador.")

# Configuração e execução do jogo
config_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\UnoRaw.cfg'
weights_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\UnoRaw_best.weights'
names_path = 'C:\\Users\\moura\\Searches\\seminario-zapzap\\models\\unoDetectorModel\\uno.names'
input_file = 'detections.txt'
output_file = 'top_class_color_pairs.txt'

with open('detections.txt', 'w') as f:
    pass
detector = UnoCardDetector(config_path, weights_path, names_path)
data_processor = DataProcessor(input_file, output_file)  # Cria a instância do DataProcessor
game = UnoGame(detector, data_processor)

game.play_game()