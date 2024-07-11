[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/49lfu3gN)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15238054&assignment_repo_type=AssignmentRepo)
# INF0417 - Computer Vision

## Seminário

### Integrantes
- 202204882 - Carlos Henrique
- 202204883 - Edward Scott
- 202204283 - Letícia Mendes
- 202202448 - Lisandra Menezes
- 202202452 - Marcos Vinicius

### [Link para o relatório](https://docs.google.com/document/d/1MbKAPjRs9rijoPWYAb_aqmyYjEI99ii4TLMICg1S3_k/edit?usp=sharing)

### [Link para o pitch](https://www.canva.com/design/DAGIxChZ-xs/RTFOvX1mD3-ZtqcXbYUciA/edit)

### Objetivo

Pretendemos construir um trapaceiro que detecta, em tempo real, as cartas e, a partir de sistemas lógicos, define suas jogadas. Para a parte de jogabilidade, utilizaremos a biblioteca RLcard, que nos auxiliará na implementação das estratégias e regras dos jogos. O projeto, portanto, envolve métodos de detecção de objetos, classificação de objetos e segmentação.

Avaliaremos nossa aplicação com base na quantidade de partidas executadas corretamente e na precisão da detecção das cartas.
Inspirados por projetos como Stopfish, buscamos desafiar a capacidade humana em jogos de cartas, ultrapassando seus limites e construindo um oponente desafiador para os jogadores de Truco e UNO. O sucesso do projeto será medido tanto pela precisão técnica quanto pela capacidade do sistema em oferecer um desafio significativo aos jogadores humanos.

### Proposta

- **Introdução**:

Com o avanço das técnicas de visão computacional e inteligência artificial, é possível criar sistemas que rivalizam com a habilidade humana em jogos complexos. Nesse contexto, nosso projeto propõe a criação de um jogador automatizado para os populares jogos de cartas Truco e UNO. Utilizando técnicas avançadas de visão computacional, pretendemos desenvolver um sistema capaz de **detectar, em tempo real**, as cartas dos jogos e, com base em bibliotecas de jogos, implementar a jogabilidade.
Para alcançar esse objetivo, empregaremos a biblioteca RICard, uma ferramenta robusta para a implementação de estratégias e regras específicas desses jogos. Nosso projeto se concentrará em métodos de **detecção de objetos, classificação de objetos e segmentação**, fundamentais para que o sistema reconheça as cartas e suas posições de forma precisa e eficiente.

A avaliação da nossa aplicação será baseada em dois critérios principais: **a quantidade de partidas executadas corretamente e a precisão na detecção das cartas**. Esses indicadores nos permitirão medir o desempenho do sistema e identificar áreas para melhorias contínuas.

Utilizamos dois conjuntos de dados para a detecção: cartas de Uno e cartas tradicionais de baralho. Esses conjuntos de dados foram obtidos no Roboflow, e o link para acessá-los está disponível nas referências.

- **Literatura**:
  - [StopFIsh](https://stockfishchess.org/)
  - [AlphaGo](https://g1.globo.com/tecnologia/noticia/alphago-inteligencia-artificial-do-google-e-aposentada-apos-vencer-melhor-jogador-de-go-do-mundo.ghtml)
  - [DeepStack vs. Pluribus](https://www.geeksforgeeks.org/deepstack-vs-pluribus-which-ai-dominates-in-poker-strategy/)
  - [SIMA](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)
- **Dataset**:
  - [CardDataset](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)
  - [UnoCardDataset](https://public.roboflow.com/object-detection/uno-cards)
  - [Github](https://github.com/Grupo-OpenCV-BR/tutoriais-tecnologia)
