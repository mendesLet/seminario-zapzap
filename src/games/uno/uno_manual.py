import random

class Carta:
    def __init__(self, cor, valor):
        self.cor = cor
        self.valor = valor
    
    def __str__(self):
        return f"{self.valor} {self.cor}"

# Definindo as cores e os valores possíveis
cores = ["Vermelho", "Amarelo", "Verde", "Azul"]
valores = [str(n) for n in range(0, 10)] + ["Pular", "Inverter", "Comprar 2"]
valores_especiais = ["Coringa", "Comprar 4"]

def criar_baralho():
    baralho = []
    for cor in cores:
        for valor in valores:
            # Cada cor tem duas cópias de cada valor (exceto 0)
            baralho.append(Carta(cor, valor))
            if valor != "0":
                baralho.append(Carta(cor, valor))
    for valor in valores_especiais:
        for _ in range(4):  # Existem quatro cartas de cada valor especial
            baralho.append(Carta("Especial", valor))
    random.shuffle(baralho)
    return baralho

# Exemplo de criação do baralho
baralho = criar_baralho()

def distribuir_cartas(baralho, num_jogadores):
    mao_jogadores = [[] for _ in range(num_jogadores)]
    for _ in range(7):  # Cada jogador recebe 7 cartas
        for mao in mao_jogadores:
            mao.append(baralho.pop())
    return mao_jogadores

# Exemplo de distribuição
num_jogadores = 4
maos = distribuir_cartas(baralho, num_jogadores)


class Uno:
    def __init__(self, num_jogadores):
        self.baralho = criar_baralho()
        self.maos = distribuir_cartas(self.baralho, num_jogadores)
        self.pilha_descarte = [self.baralho.pop()]
        self.direcao = 1  # 1 para horário, -1 para anti-horário
        self.jogador_atual = 0
    
    def mostrar_estado(self):
        print(f"Jogador atual: {self.jogador_atual + 1}")
        print(f"Carta na pilha de descarte: {self.pilha_descarte[-1]}")
        print(f"Mão do jogador {self.jogador_atual + 1}: {[str(carta) for carta in self.maos[self.jogador_atual]]}")

    def jogar_carta(self, carta):
        ultima_carta = self.pilha_descarte[-1]
        if carta.cor == ultima_carta.cor or carta.valor == ultima_carta.valor or carta.cor == "Especial":
            self.pilha_descarte.append(carta)
            self.maos[self.jogador_atual].remove(carta)
            
            if carta.valor == "Pular":
                self.proximo_turno()
            elif carta.valor == "Inverter":
                self.direcao *= -1
            elif carta.valor == "Comprar 2":
                self.proximo_turno()
                self.jogador_atual = (self.jogador_atual + self.direcao) % len(self.maos)
                for _ in range(2):
                    self.comprar_carta()
            elif carta.valor == "Comprar 4":
                self.proximo_turno()
                self.jogador_atual = (self.jogador_atual + self.direcao) % len(self.maos)
                for _ in range(4):
                    self.comprar_carta()
                self.mudar_cor()
            elif carta.valor == "Coringa":
                self.mudar_cor()
            
            return True
        return False

    def comprar_carta(self):
        if not self.baralho:
            self.baralho = self.pilha_descarte[:-1]
            self.pilha_descarte = [self.pilha_descarte[-1]]
            random.shuffle(self.baralho)
        carta = self.baralho.pop()
        self.maos[self.jogador_atual].append(carta)

    def proximo_turno(self):
        self.jogador_atual = (self.jogador_atual + self.direcao) % len(self.maos)
    
    def verificar_vencedor(self):
        for i, mao in enumerate(self.maos):
            if not mao:
                return i
        return -1
    
    def mudar_cor(self):
        while True:
            nova_cor = input("Escolha uma cor (Vermelho, Amarelo, Verde, Azul): ").capitalize()
            if nova_cor in cores:
                self.pilha_descarte[-1].cor = nova_cor
                break
            else:
                print("Cor inválida. Tente novamente.")


def jogar_uno(num_jogadores):
    uno = Uno(num_jogadores)

    while True:
        uno.mostrar_estado()
        jogador = uno.jogador_atual
        print(f"Jogador {jogador + 1}, é sua vez!")
        
        # Mostra as cartas na mão do jogador atual
        print("Suas cartas:")
        for idx, carta in enumerate(uno.maos[jogador]):
            print(f"{idx + 1}: {carta}")
        
        # Jogador escolhe uma carta ou compra
        escolha = input("Escolha uma carta para jogar (número) ou digite 'comprar' para pegar uma carta: ")
        
        if escolha.lower() == "comprar":
            uno.comprar_carta()
        else:
            try:
                indice = int(escolha) - 1
                if indice < 0 or indice >= len(uno.maos[jogador]):
                    raise ValueError("Índice inválido.")
                carta_para_jogar = uno.maos[jogador][indice]
                
                if not uno.jogar_carta(carta_para_jogar):
                    print("Você não pode jogar essa carta. Tente novamente.")
                    continue
            except ValueError as e:
                print(f"Entrada inválida: {e}. Tente novamente.")
                continue
        
        # Verifica se há um vencedor
        vencedor = uno.verificar_vencedor()
        if vencedor != -1:
            print(f"Jogador {vencedor + 1} venceu!")
            break
        
        # Passa para o próximo turno
        uno.proximo_turno()


if __name__ == "__main__":
    num_jogadores = int(input("Quantos jogadores? "))
    jogar_uno(num_jogadores)
