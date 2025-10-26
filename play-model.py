import numpy as np
import pygame
import random
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import LeakyReLU # type: ignore


# pygame template

# windows size
WIDTH = 600
HEIGHT = 600
FPS = 20

# colors RGB
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODEL_NO = 100 # En son eğitilmiş model nosu girilmeli
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

class Tick(pygame.sprite.Sprite):
    def __init__(self,t):
        pygame.sprite.Sprite.__init__(self)
        self.t = t
        self.image = pygame.Surface((15,150))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.radius = 5

        pygame.draw.circle(self.image,RED,self.rect.center,self.radius)

        if t == 1:
            self.rect.centery = HEIGHT // 2
            self.rect.centerx = 10
            self.speedy = 0

        if t == 2:
            self.rect.centery = HEIGHT // 2
            self.rect.centerx = WIDTH - 10
            self.speedy = 0

    def update(self,action):
        self.speedy = 0
        if action == 0:
            self.speedy = -18
        elif action == 1:
            self.speedy = 18
        self.rect.y  += self.speedy

        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT

    def getCoordinate(self):
        return (self.rect.x, self.rect.y)

class Ball(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((10,10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image,RED,self.rect.center, self.radius)

        self.rect.centerx = WIDTH // 2
        self.rect.centery = HEIGHT // 2

        alpha = random.randrange(1,89)
        dif = 90-alpha
        self.speedx = 12*np.cos(np.deg2rad(alpha))
        self.speedy = 12*np.cos(np.deg2rad(dif))

        self.speedx *= random.choice([-1.01,1.01])
        self.speedy *= random.choice([-1.01,1.01])

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



class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.all_sprite = pygame.sprite.Group()
        self.ticks = pygame.sprite.Group()
        self.balls = pygame.sprite.Group()

        self.t1 = Tick(1)
        self.t2 = Tick(2)
        self.ball = Ball()

        self.all_sprite.add(self.t1)
        self.all_sprite.add(self.t2)
        self.all_sprite.add(self.ball)

        self.balls.add(self.ball)

    def findDistance(self,a,b):
        return a - b  # Yönlü fark (pozitif/negatif)

    def step(self,action1,action2):
        self.t1.update(action1)
        self.t2.update(action2)
        self.ball.update()

        state_list_1 = []
        state_list_2 = []

        next_t1_state = self.t1.getCoordinate()
        next_t2_state = self.t2.getCoordinate()
        next_ball = self.ball.getCoordinate()

        state_list_1.append(self.findDistance(next_t1_state[0],next_ball[0]))
        state_list_1.append(self.findDistance(next_t1_state[1],next_ball[1]))

        state_list_2.append(self.findDistance(next_t2_state[0], next_ball[0]))
        state_list_2.append(self.findDistance(next_t2_state[1], next_ball[1]))

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
        return [state_list_1],[state_list_2]

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

        state_list_1 = []
        state_list_2 = []
        t1_state = self.t1.getCoordinate()
        t2_state = self.t2.getCoordinate()
        ball_state = self.ball.getCoordinate()

        state_list_1.append(self.findDistance(t1_state[0],ball_state[0]))
        state_list_1.append(self.findDistance(t1_state[1],ball_state[1]))

        state_list_2.append(self.findDistance(t2_state[0],ball_state[0]))
        state_list_2.append(self.findDistance(t2_state[1],ball_state[1]))

        return [state_list_1],[state_list_2]

    def loadModel(self, t, no):
        return load_model(f"models/agent{t}_ep{no}.keras",
                        custom_objects={"LeakyReLU": LeakyReLU},
                        compile=False)

# initialize pygame and creat window
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("RL Game")
clock = pygame.time.Clock()
env = Env()
state1, state2 = env.initialStates()
model1 = env.loadModel(1,MODEL_NO)  # Agent 1'in modeli
model2 = env.loadModel(2,MODEL_NO)  # Agent 2'nin modeli
# game loop
running = True
while running:

    # keep loop running at the right speed
    clock.tick(FPS)

    # process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # update

    q1 = model1.predict(np.array(state1), verbose=0)[0]
    q2 = model2.predict(np.array(state2), verbose=0)[0]
    action1 = int(np.argmax(q1))
    action2 = int(np.argmax(q2))
    
    # Debug: Action değerlerini yazdır
    if random.random() < 0.01:  # %1 ihtimalle yazdır
        print(f"Agent1 Q-values: {q1}, Action: {action1}")
        print(f"Agent2 Q-values: {q2}, Action: {action2}")

    state1, state2 = env.step(action1, action2)

    fail = env.ball.didFail()
    if fail is not None:
        # round bitti; yeniden başlatmak istersen:
        state1, state2 = env.initialStates()

    # drow and render (show)
    screen.fill(BLACK)
    env.all_sprite.draw(screen)
    # after drawing flip display
    pygame.display.flip()

pygame.quit()