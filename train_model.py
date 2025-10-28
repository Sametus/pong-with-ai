import os
import re
import glob
import gzip
import pickle
import warnings
warnings.filterwarnings("ignore")

import pygame
import random
import gym
import numpy as np
from collections import deque

from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Input, LeakyReLU # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


WIDTH = 600
HEIGHT = 600
FPS = 20

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)


MODELS_DIR = "models"


def save_agent_state(agent, path):
    state = {
        "epsilon": agent.epsilon,
        "memory": list(agent.memory),  # deque ==> list (serialize)
    }
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def load_agent_state(agent, path):
    with gzip.open(path, "rb") as f:
        state = pickle.load(f)
    agent.epsilon = state.get("epsilon", agent.epsilon)
    agent.memory  = deque(state.get("memory", []), maxlen=agent.memory.maxlen)

def latest_episode(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    nums = []
    for p in files:
        m = re.search(r"_ep(\d+)\.keras$", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else None


class Tick(pygame.sprite.Sprite):
    def __init__(self, t):
        pygame.sprite.Sprite.__init__(self)
        self.t = t
        self.image = pygame.Surface((15, 150))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        if t == 1:
            self.rect.centery = HEIGHT // 2
            self.rect.centerx = 10
            self.speedy = 0
        if t == 2:
            self.rect.centery = HEIGHT // 2
            self.rect.centerx = WIDTH - 10
            self.speedy = 0

    def update(self, action):
        self.speedy = 0
        if action == 0:
            self.speedy = -18
        elif action == 1:
            self.speedy = 18
        self.rect.y += self.speedy
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT

    def getCoordinate(self):
        return (self.rect.x, self.rect.y)


class Ball(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)

        self.rect.centerx = WIDTH / 2
        self.rect.centery = HEIGHT / 2

        alpha = random.randrange(1, 89)
        dif = 90 - alpha

        self.speedx = 12 * np.cos(np.deg2rad(alpha))
        self.speedy = 12 * np.cos(np.deg2rad(dif))

        self.speedx *= random.choice([-1.01, 1.01])
        self.speedy *= random.choice([-1.01, 1.01])

    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy

        if self.rect.top < 0:
            self.speedy *= -1
        if self.rect.bottom > HEIGHT:
            self.speedy *= -1

    def collision(self):
        self.speedx *= -1

    def getCoordinate(self):
        return (self.rect.x, self.rect.y)

    def didFail(self):
        if self.rect.right > WIDTH:
            return "right"
        if self.rect.left < 0:
            return "left"
        return None


class DQLAgent:
    def __init__(self):
        self.state_size = 2  # [tick.x - ball.x, tick.y - ball.y]
        self.action_size = 3

        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=3000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(32),
            LeakyReLU(alpha=0.01),
            Dense(32),
            LeakyReLU(alpha=0.01),
            Dense(self.action_size, activation="linear"),
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        s = np.asarray(state, dtype=np.float32)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(s, verbose=0)
        return int(np.argmax(q[0]))

    def replay(self, batch_size):
        "vectorized replay method"
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch, dtype=object)

        not_done_indices = np.where(minibatch[:, 4] == False)[0]
        y = np.copy(minibatch[:, 2]).astype(np.float32)

        if not_done_indices.size > 0:
            ns = np.vstack(minibatch[:, 3]).astype(np.float32)
            predict_sprime = self.model.predict(ns, verbose=0)         
            predict_sprime_target = predict_sprime                     
            best_next = np.argmax(predict_sprime[not_done_indices, :], axis=1)
            y[not_done_indices] += self.gamma * predict_sprime_target[not_done_indices, best_next]

        actions = np.array(minibatch[:, 1], dtype=int)
        X = np.vstack(minibatch[:, 0]).astype(np.float32)
        y_target = self.model.predict(X, verbose=0)
        y_target[np.arange(batch_size), actions] = y

        self.model.fit(X, y_target, epochs=1, verbose=0, batch_size=batch_size)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()

        self.ticks = pygame.sprite.Group()
        self.t1 = Tick(1)
        self.t2 = Tick(2)

        self.balls = pygame.sprite.Group()
        self.ball = Ball()

        self.all_sprite.add(self.t1)
        self.all_sprite.add(self.t2)
        self.all_sprite.add(self.ball)

        self.ticks.add(self.t1)
        self.ticks.add(self.t2)
        self.balls.add(self.ball)

        self.reward_1 = 0
        self.reward_2 = 0
        self.total_reward_1 = 0
        self.total_reward_2 = 0
        self.done_1 = False
        self.done_2 = False

        self.agent1 = DQLAgent()
        self.agent2 = DQLAgent()

    def findDistance(self, a, b):
        return a - b

    def step(self, action1, action2):
        self.t1.update(action1)
        self.t2.update(action2)
        self.ball.update()

        if self.ball.rect.colliderect(self.t2.rect):
            if self.ball.speedx > 0:
                self.ball.rect.right = self.t2.rect.left
            else:
                self.ball.rect.left = self.t2.rect.right
            self.ball.collision()

        if self.ball.rect.colliderect(self.t1.rect):
            if self.ball.speedx < 0:
                self.ball.rect.left = self.t1.rect.right
            else:
                self.ball.rect.right = self.t1.rect.left
            self.ball.collision()

        state_list_1 = []
        state_list_2 = []

        next_t1_state = self.t1.getCoordinate()
        next_t2_state = self.t2.getCoordinate()
        next_ball_state = self.ball.getCoordinate()

        state_list_1.append(self.findDistance(next_t1_state[0], next_ball_state[0]))
        state_list_1.append(self.findDistance(next_t1_state[1], next_ball_state[1]))

        state_list_2.append(self.findDistance(next_t2_state[0], next_ball_state[0]))
        state_list_2.append(self.findDistance(next_t2_state[1], next_ball_state[1]))

        return [state_list_1], [state_list_2]

    def initialStates(self):

        self.all_sprite = pygame.sprite.Group()
        self.ticks = pygame.sprite.Group()
        self.t1 = Tick(1)
        self.t2 = Tick(2)

        self.balls = pygame.sprite.Group()
        self.ball = Ball()

        self.all_sprite.add(self.t1)
        self.all_sprite.add(self.t2)
        self.all_sprite.add(self.ball)

        self.ticks.add(self.t1)
        self.ticks.add(self.t2)
        self.balls.add(self.ball)

        self.reward_1 = 0
        self.total_reward_1 = 0
        self.done_1 = False
        self.reward_2 = 0
        self.total_reward_2 = 0
        self.done_2 = False

        state_list_1 = []
        state_list_2 = []
        t1_state = self.t1.getCoordinate()
        t2_state = self.t2.getCoordinate()
        ball_state = self.ball.getCoordinate()

        state_list_1.append(self.findDistance(t1_state[0], ball_state[0]))
        state_list_1.append(self.findDistance(t1_state[1], ball_state[1]))

        state_list_2.append(self.findDistance(t2_state[0], ball_state[0]))
        state_list_2.append(self.findDistance(t2_state[1], ball_state[1]))

        return [state_list_1], [state_list_2]

    def run(self):
        s1, s2 = self.initialStates()
        state_1 = s1
        state_2 = s2
        batch_size = 32
        running = True
        while running:
            self.reward_1 = 2
            self.reward_2 = 2
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # eylemler
            a1 = self.agent1.act(state_1)
            a2 = self.agent2.act(state_2)

            next_state_1, next_state_2 = self.step(a1, a2)

            self.total_reward_1 += self.reward_1
            self.total_reward_2 += self.reward_2

            # bitiş kontrolü
            if self.ball.didFail() == "right":
                self.reward_2 = -150
                self.total_reward_2 += self.reward_2
                self.done_2 = True
                running = False
                print(f"Total Reward - 2: {self.total_reward_2}")

            if self.ball.didFail() == "left":
                self.reward_1 = -150
                self.total_reward_1 += self.reward_1
                self.done_1 = True
                running = False
                print(f"Total Reward - 1: {self.total_reward_1}")

            # çizim
            screen.fill(BLACK)
            self.all_sprite.draw(screen)

            # deneyim kaydı
            self.agent1.remember(state_1, a1, self.reward_1, next_state_1, self.done_1)
            self.agent2.remember(state_2, a2, self.reward_2, next_state_2, self.done_2)

            state_1 = next_state_1
            state_2 = next_state_2

            # eğitim
            self.agent1.replay(batch_size=batch_size)
            self.agent1.adaptiveEGreedy()

            self.agent2.replay(batch_size=batch_size)
            self.agent2.adaptiveEGreedy()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = Env()
    liste1, liste2 = [], []


    last1 = latest_episode(os.path.join(MODELS_DIR, "agent1_ep*.keras"))
    last2 = latest_episode(os.path.join(MODELS_DIR, "agent2_ep*.keras"))

    if last1 is not None and last2 is not None:
        t_resume = min(last1, last2)

        env.agent1.model = load_model(
            os.path.join(MODELS_DIR, f"agent1_ep{t_resume}.keras"),
            custom_objects={"LeakyReLU": LeakyReLU}
        )
        env.agent2.model = load_model(
            os.path.join(MODELS_DIR, f"agent2_ep{t_resume}.keras"),
            custom_objects={"LeakyReLU": LeakyReLU}
        )

        try:
            load_agent_state(env.agent1, os.path.join(MODELS_DIR, f"agent1_ep{t_resume}.state.gz"))
            load_agent_state(env.agent2, os.path.join(MODELS_DIR, f"agent2_ep{t_resume}.state.gz"))
            print(f"[RESUME] {t_resume}. bölümden devam ediliyor.")
        except FileNotFoundError:
            print(f"[RESUME] Model bulundu, fakat state dosyası yok. {t_resume+1}. bölümden optimizer ile devam.")
        start_ep = t_resume + 1
    else:
        print("[RESUME] Checkpoint bulunamadı; 1. bölümden başlanıyor.")
        start_ep = 1

    EPISODES = 1000

    for t in range(start_ep, EPISODES + 1):
        print(f"Episode: {t}")

        # pygame init/loop
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()

        env.run()

        liste1.append(env.total_reward_1)
        liste2.append(env.total_reward_2)

        # checkpoint
        if t % 50 == 0:
            env.agent1.model.save(os.path.join(MODELS_DIR, f"agent1_ep{t}.keras"))
            env.agent2.model.save(os.path.join(MODELS_DIR, f"agent2_ep{t}.keras"))
            save_agent_state(env.agent1, os.path.join(MODELS_DIR, f"agent1_ep{t}.state.gz"))
            save_agent_state(env.agent2, os.path.join(MODELS_DIR, f"agent2_ep{t}.state.gz"))


    env.agent1.model.save(os.path.join(MODELS_DIR, f"agent1_ep{t}.keras"))
    env.agent2.model.save(os.path.join(MODELS_DIR, f"agent2_ep{t}.keras"))
    save_agent_state(env.agent1, os.path.join(MODELS_DIR, f"agent1_ep{t}.state.gz"))
    save_agent_state(env.agent2, os.path.join(MODELS_DIR, f"agent2_ep{t}.state.gz"))

    print("Training completed to models folder")
