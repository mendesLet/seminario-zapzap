import numpy as np
from collections import defaultdict
import threading
import random
import requests
import time
import sys

class JogoCartasRL:
    def __init__(self, num_jogos):
        self.cartas = ['4P', 'QP', 'JP', 'KP', 'AP', '2P', '3P',    # Paus
                       '7C', 'QC', 'JC', 'KC', 'AC', '2C', '3C',    # Copas
                       'QE', 'JE', 'KE', 'AE', '2E', '3E',          # Espadas
                       '7O', 'QO', 'JO', 'KO', 'AO', '2O', '3O']    # Ouros

        self.valores = {'4P': 14, 'QP': 5, 'JP': 6, 'KP': 7, 'AP': 8, '2P': 9, '3P': 10,
                        '7C': 13, 'QC': 5, 'JC': 6, 'KC': 7, 'AC': 8, '2C': 9, '3C': 10,
                        'QE': 5, 'JE': 6, 'KE': 7, 'AE': 12, '2E': 9, '3E': 10,
                        '7O': 11, 'QO': 5, 'JO': 6, 'KO': 7, 'AO': 8, '2O': 9, '3O': 10}

        self.epsilon = 3
        self.epsilon_min = 2
        self.epsilon_decay = 0.80
        self.gamma = 0.59

        self.Q1 = defaultdict(lambda: np.zeros(len(self.cartas) + 1))
        self.Q2 = defaultdict(lambda: np.zeros(len(self.cartas) + 1))

        self.num_jogos = num_jogos
        self.recompensas_agente1 = []
        self.recompensas_agente2 = []
        self.vitorias_agente1 = 0
        self.vitorias_agente2 = 0
        self.decisoes_agente1 = defaultdict(int)
        self.decisoes_agente2 = defaultdict(int)

        self.latest_prediction = None
        self.prediction_event = threading.Event()

    def initialize_game_agents(self):
        agente1_hand = random.sample(self.cartas, k=3)
        agente2_hand = random.sample(self.cartas, k=3)
        mesa = []
        aposta = 1
        vez = -1
        return agente1_hand, agente2_hand, mesa, vez, aposta

    def initialize_game(self):

        mesa = []
        aposta = 1
        vez = 1
        return mesa, vez, aposta

    def choose_action(self, Q, hand, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(len(hand) + 1)
        else:
            q_values = Q[state][:len(hand) + 1]
            return np.argmax(q_values)

    def take_action(self, hand, mesa, action):
        if action <= len(hand):
            card = hand[action]
            hand.remove(card)
            mesa.append(card)

    def take_action_player(self, mesa, action):
        
            card = action
            mesa.append(card)

    def determinar_vencedor(self, mesa):
        agente1_pontos = sum(self.valores.get(carta, 0) for carta in mesa if mesa.index(carta) % 2 == 0)
        agente2_pontos = sum(self.valores.get(carta, 0) for carta in mesa if mesa.index(carta) % 2 != 0)

        if agente1_pontos > agente2_pontos:
            return 1
        else:
            return -1

    def processar_blefe(self, agente_blefe, agente_oponente, aposta):
        resposta = np.random.choice(['aceitar', 'rejeitar'], p=[0.5, 0.5])
        if resposta == 'aceitar':
            aposta *= 2
            return False, aposta
        else:
            return True, aposta

    def train_agents(self):
        for episode in range(self.num_jogos):
            agente1_hand, agente2_hand, mesa, vez, aposta = self.initialize_game_agents()
            total_reward1 = 0
            total_reward2 = 0
            done = False

            while not done:
                state = (tuple(sorted(agente1_hand)), tuple(sorted(agente2_hand)), tuple(mesa), vez)

                if vez == 1:
                    action = self.choose_action(self.Q1, agente1_hand, state, self.epsilon)
                    self.decisoes_agente1[action] += 1
                    if action == len(agente1_hand):
                        done, aposta = self.processar_blefe(agente1_hand, agente2_hand, aposta)
                        reward = aposta if done else 0
                        total_reward1 += reward
                        next_state = (tuple(sorted(agente1_hand)), tuple(sorted(agente2_hand)), tuple(mesa), vez)
                        self.Q1[state][action] = self.Q1[state][action] + self.gamma * (
                                    reward + np.max(self.Q1[next_state]) - self.Q1[state][action])
                    else:
                        self.take_action(agente1_hand, mesa, action)
                else:
                    action = self.choose_action(self.Q2, agente2_hand, state, self.epsilon)
                    self.decisoes_agente2[action] += 1
                    if action == len(agente2_hand):
                        done, aposta = self.processar_blefe(agente2_hand, agente1_hand, aposta)
                        reward = aposta if done else 0
                        total_reward2 += reward
                        next_state = (tuple(sorted(agente1_hand)), tuple(sorted(agente2_hand)), tuple(mesa), vez)
                        self.Q2[state][action] = self.Q2[state][action] + self.gamma * (
                                    reward + np.max(self.Q2[next_state]) - self.Q2[state][action])
                    else:
                        self.take_action(agente2_hand, mesa, action)

                if len(agente1_hand) == 0 or len(agente2_hand) == 0:
                    reward = self.determinar_vencedor(mesa) * aposta
                    if vez == 1:
                        total_reward1 += reward
                        next_state = (tuple(sorted(agente1_hand)), tuple(sorted(agente2_hand)), tuple(mesa), vez)
                        self.Q1[state][action] = self.Q1[state][action] + self.gamma * (
                                    reward + np.max(self.Q1[next_state]) - self.Q1[state][action])
                    else:
                        total_reward2 += -reward
                        next_state = (tuple(sorted(agente1_hand)), tuple(sorted(agente2_hand)), tuple(mesa), vez)
                        self.Q2[state][action] = self.Q2[state][action] + self.gamma * (
                                    -reward + np.max(self.Q2[next_state]) - self.Q2[state][action])
                    done = True
                else:
                    vez *= -1

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.recompensas_agente1.append(total_reward1)
            self.recompensas_agente2.append(total_reward2)

            if self.determinar_vencedor(mesa) == 1:
                self.vitorias_agente1 += 1
            else:
                self.vitorias_agente2 += 1

            if episode % 50 == 0:
                print(f'Episódio {episode}, Recompensa Total Agente 1: {total_reward1}, Recompensa Total Agente 2: {total_reward2}, Epsilon: {self.epsilon}')

    def calcular_metricas_finais(self):
        total_jogos = len(self.recompensas_agente1) + len(self.recompensas_agente2)
        acuracia_agente1 = self.decisoes_agente1[np.argmax(list(self.decisoes_agente1.values()))] / total_jogos
        acuracia_agente2 = self.decisoes_agente2[np.argmax(list(self.decisoes_agente2.values()))] / total_jogos
        media_recompensa_agente1 = np.mean(self.recompensas_agente1)
        media_recompensa_agente2 = np.mean(self.recompensas_agente2)
        media_rodadas_jogo = (len(self.recompensas_agente1) + len(self.recompensas_agente2)) / total_jogos

        print("\n--- Métricas Finais ---")
        print(f"Total de jogos ganhos por Agente 1: {self.vitorias_agente1}")
        print(f"Total de jogos ganhos por Agente 2: {self.vitorias_agente2}")
        print(f"Acurácia das decisões de Agente 1: {acuracia_agente1}")
        print(f"Acurácia das decisões de Agente 2: {acuracia_agente2}")
        print(f"Recompensa média por jogo de Agente 1: {media_recompensa_agente1}")
        print(f"Recompensa média por jogo de Agente 2: {media_recompensa_agente2}")
        print(f"Número médio de rodadas por jogo: {media_rodadas_jogo}")

    def fetch_latest_prediction(self):
        while True:
            try:
                response = requests.get('http://localhost:5000/latest_prediction')
                if response.status_code == 200:
                    prediction = response.json().get('prediction')
                    if prediction:
                        self.latest_prediction = prediction
                        self.prediction_event.set()
                time.sleep(1)
            except Exception as e:
                print(f'Error fetching prediction: {e}')
                time.sleep(5)  # Retry after a delay

    def start_prediction_thread(self):
        prediction_thread = threading.Thread(target=self.fetch_latest_prediction)
        prediction_thread.start()

    def play_with_agent(self):
        self.start_prediction_thread()

        # Wait for the initial hand to be fetched
        self.prediction_event.wait()
        hand = self.latest_prediction
        while len(hand) != 3:
            self.prediction_event.wait()
            hand = self.latest_prediction

        print(hand)

        self.pt_ag2 = 0
        self.pt_ag1 = 0

        for round_num in range(1, 4):  # Loop for 3 rounds
            print(f"--- Rodada {round_num} ---")
            mesa, vez, aposta = self.initialize_game()

            while True:
                if vez == 1:
                    
                    print("\n--- Sua Vez ---")
                    acao = []
                    print(f"Mesa: {mesa}")

                    # Fetch the latest prediction when available
                    self.prediction_event.wait()
                    acao = self.latest_prediction
                    while not acao or len(acao) != 1:
                        self.prediction_event.wait()
                        acao = self.latest_prediction
                        self.prediction_event.clear()

                    card = acao[0]
                    mesa.append(card)
                    print(f"Você jogou {acao}.")
                    vez = -1  # Atualiza vez após jogar
                
                elif vez == -1:
                    
                    print("\n--- Vez do Agente ---")
                    state = (tuple(sorted([])), tuple(sorted(hand)), tuple(mesa), vez)
                    action = self.choose_action(self.Q2, hand, state, self.epsilon)
                    while action >= len(hand):
                        action = self.choose_action(self.Q2, hand, state, self.epsilon)
                    acao = hand[action]
                    self.take_action(hand, mesa, action)
                    print(f"Agente jogou {acao}.")
                    time.sleep(10)
                    vez = 1  # Atualiza vez após agente jogar

                    vencedor = self.determinar_vencedor(mesa)
                    if vencedor == 1:
                        print("Você venceu a rodada!")
                        self.pt_ag1 += 1
                        break
                    elif vencedor == -1:
                        print("O Agente venceu a rodada!")
                        self.pt_ag2 += 1
                        break

        print(f"Pontuação final: Agente 2 = {self.pt_ag2}, Você = {self.pt_ag1}")
        print("O jogo acabou!")
        if self.pt_ag1 > self.pt_ag2:
            print("Você é o vencedor do jogo!")
        elif self.pt_ag2 > self.pt_ag1:
            print("Agente 2 é o vencedor do jogo!")
        else:
            print("O jogo terminou empatado!")



# Exemplo de uso da classe
if __name__ == '__main__':
    jogo = JogoCartasRL(num_jogos=100)
    jogo.train_agents()
    jogo.calcular_metricas_finais()
    jogo.play_with_agent()
