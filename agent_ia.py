import pygame
import numpy as np
import random
import time

# Configurações
GRID_SIZE = 8
CELL_SIZE = 60
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
START = (0, 0)
GOAL = (7, 7)
OBSTACLES = [
    (0,1), (1,1), (2,1), (4,1), (4,2),
    (3,2), (3,4), (3,5), (4,5), (5,5),
    (5,4), (6,6), (7,6), (1,3), (2,4),
    (0,5), (2,6), (0,7)
]

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # cima, baixo, esquerda, direita

# Parâmetros do Q-Learning
ALPHA = 1.0
GAMMA = 0.9
EPSILON = 0.8
EPISODES = 80
MAX_STEPS = 256

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
PURPLE = (180, 0, 255)
ORANGE = (255, 165, 0)

# Q-tables separadas
q_agent1 = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
q_agent2 = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Funções auxiliares
def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def get_next_state(state, action):
    dx, dy = ACTIONS[action]
    next_state = (state[0] + dx, state[1] + dy)
    return next_state if is_valid(next_state) else state

def get_reward(state):
    if state == GOAL:
        return 100
    elif state in OBSTACLES:
        return -10
    else:
        return -2

def draw_grid(screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (x, y) in OBSTACLES:
                color = BLACK
            elif (x, y) == START:
                color = GREEN
            elif (x, y) == GOAL:
                color = RED
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GREY, rect, 1)

def draw_agent(screen, pos, color):
    rect = pygame.Rect(pos[0] * CELL_SIZE + 10, pos[1] * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, color, rect)

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# -------------------------------------------
# Treinamento com dois agentes
# -------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treinamento: Corrida de Dois Agentes")
clock = pygame.time.Clock()

for episode in range(EPISODES):
    print(f"Episódio {episode + 1}/{EPISODES}")
    state1 = START
    state2 = START
    winner = None

    for step in range(MAX_STEPS):
        process_events()

        # Escolher ação para cada agente
        action1 = random.randint(0, 3) if random.random() < EPSILON else np.argmax(q_agent1[state1[0], state1[1]])
        action2 = random.randint(0, 3) if random.random() < EPSILON else np.argmax(q_agent2[state2[0], state2[1]])

        # Próximos estados
        next1 = get_next_state(state1, action1)
        next2 = get_next_state(state2, action2)

        # Recompensas
        reward1 = get_reward(next1)
        reward2 = get_reward(next2)

        # Atualizar Q-tables
        q_agent1[state1[0], state1[1], action1] += ALPHA * (reward1 + GAMMA * np.max(q_agent1[next1[0], next1[1]]) - q_agent1[state1[0], state1[1], action1])
        q_agent2[state2[0], state2[1], action2] += ALPHA * (reward2 + GAMMA * np.max(q_agent2[next2[0], next2[1]]) - q_agent2[state2[0], state2[1], action2])

        # Atualizar posições
        state1, state2 = next1, next2

        # Verificar vencedor
        if state1 == GOAL:
            winner = "Agente Azul"
            break
        if state2 == GOAL:
            winner = "Agente Roxo"
            break

        # Visualização
        screen.fill(WHITE)
        draw_grid(screen)
        draw_agent(screen, state1, BLUE)
        draw_agent(screen, state2, PURPLE)
        pygame.display.flip()
        clock.tick(60)

    if winner:
        print(f"🏁 {winner} venceu o episódio {episode + 1}!")
        time.sleep(0.6)

    EPSILON = max(0.1, EPSILON * 0.97)  # Reduz exploração gradualmente

print("Treinamento concluído!")
time.sleep(1)
pygame.display.quit()

# -------------------------------------------
# Execução dos agentes treinados (corrida final)
# -------------------------------------------
pygame.display.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Corrida Final dos Agentes Treinados")
clock = pygame.time.Clock()

pos1 = START
pos2 = START
path1 = [pos1]
path2 = [pos2]
winner = None

while True:
    process_events()

    screen.fill(WHITE)
    draw_grid(screen)
    draw_agent(screen, pos1, BLUE)
    draw_agent(screen, pos2, PURPLE)
    pygame.display.flip()
    clock.tick(10)

    if not winner:
        if pos1 != GOAL:
            pos1 = get_next_state(pos1, np.argmax(q_agent1[pos1[0], pos1[1]]))
            path1.append(pos1)
        if pos2 != GOAL:
            pos2 = get_next_state(pos2, np.argmax(q_agent2[pos2[0], pos2[1]]))
            path2.append(pos2)

        if pos1 == GOAL and not winner:
            winner = "Agente Azul"
        elif pos2 == GOAL and not winner:
            winner = "Agente Roxo"

    else:
        print(f"\n🏆 {winner} venceu a corrida final!")
        time.sleep(2)
        pygame.quit()
        break
