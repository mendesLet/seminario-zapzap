import os
from dotenv import load_dotenv
import google.generativeai as genai

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave da API da variável de ambiente
gemini_api_key = os.getenv("GEMINI_API_KEY")

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
    generation_config=generation_config,
    system_instruction="""Você é um dos jogadores de uma partida de UNO, está jogando contra outro jogador.
        Sua mão: {hand}
        Topo da pilha de descarte: {discard_pile}
        Usando a carta da pilha de descarte, construa a melhor estratégia para eliminar o mais rápido possível todas suas cartas, respeitando as regras oficiais do UNO.

        A saída deve ser no seguinte formato:
        ##Exemplos##

        Output: Jogue as cartas 7 verde, 7 vermelho e 7 azul juntas.
        Output: Jogue a Comprar 2 Verde."""
)

class UnoGame:
    def __init__(self):
        self.player_hand = []  # Inicializar a mão do jogador
        self.discard_pile = []  # Inicializar a pilha de descarte

    def update_hand(self, new_hand):
        self.player_hand = new_hand

    def update_discard_pile(self, new_top_card):
        if not self.discard_pile or self.discard_pile[-1] != new_top_card:
            self.discard_pile.append(new_top_card)

    def uno_assistant(self, hand, discard_pile):
        prompt = f"""
        hand = {", ".join(hand)}
        discard_pile = {discard_pile}
        """
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [prompt]},
            ]
        )
        
        response = chat_session.send_message(prompt)
        
        return response.text.strip()

    def player_turn(self):
        # Implementação da lógica do turno do jogador
        print("Turno do jogador")
        # O jogador faz sua jogada manualmente
        # Atualizar manualmente para testes
        new_hand = input("Digite sua mão atualizada separada por vírgulas: ").split(',')
        self.update_hand(new_hand)

        new_top_card = input("Digite a carta no topo da pilha de descarte: ")
        self.update_discard_pile(new_top_card)

    def assistant_turn(self):
        # Implementação da lógica do turno do assistente usando LLM
        print("Turno do assistente")
        decision = self.uno_assistant(self.player_hand, self.discard_pile[-1])
        print(f"Assistente decide: {decision}")
        # Para fins de teste, não atualizamos a mão e a pilha de descarte automaticamente
        # Isso pode ser atualizado manualmente após a decisão do assistente

    def play_game(self):
        # Lógica para alternar turnos
        while not self.is_game_over():
            self.player_turn()
            self.assistant_turn()

    def is_game_over(self):
        # Código para verificar se o jogo terminou
        return len(self.player_hand) == 0

# Inicializando o jogo
game = UnoGame()

# Atualizar a mão inicial e a pilha de descarte para testes
initial_hand = input("Digite sua mão inicial separada por vírgulas: ").split(',')
game.update_hand(initial_hand)

initial_discard = input("Digite a carta inicial no topo da pilha de descarte: ")
game.update_discard_pile(initial_discard)

# Começando o jogo
game.play_game()
