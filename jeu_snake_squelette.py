"""
jeu_snake_squelette.py — SQUELETTE ÉTUDIANT (Snake)
====================================================
Le code tourne déjà (fenêtre + entraînement), MAIS tant que la récompense n'est
pas définie l'agent n'apprend rien : c'est justement l'exercice de la séance.

À COMPLÉTER (cherchez les  # TODO) :
  1) la RÉCOMPENSE dans step()  ............... le cœur du MDP
  2) l'ENTRAÎNEMENT (algo + hyperparamètres) ... en bas du fichier
  (BONUS) enrichir l'OBSERVATION dans _obs()

Lancer :  python jeu_snake_squelette.py
Dépendances :  pip install gymnasium stable-baselines3 torch pygame numpy
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import gymnasium as gym
from gymnasium import spaces

TIMESTEPS = 150_000          # temps d'entraînement (baissez pour une démo rapide)
GRID      = 12               # taille de la grille
CELL      = 32               # taille d'une case à l'écran (pixels)
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # droite, bas, gauche, haut


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, grid=GRID):
        super().__init__()
        self.grid = grid
        self.action_space = spaces.Discrete(3)                   # 0=tout droit, 1=tourner à droite, 2=à gauche
        self.observation_space = spaces.Box(0.0, 1.0, (11,), np.float32)

    def _danger(self, pt):
        x, y = pt
        return x < 0 or x >= self.grid or y < 0 or y >= self.grid or (x, y) in self.snake

    def _obs(self):
        # 11 "capteurs" FOURNIS : danger devant/droite/gauche, direction, position de la pomme.
        # (BONUS : vous pouvez ajouter des features ici — pensez à ajuster la taille (11,) ci-dessus.)
        hx, hy = self.snake[0]; dx, dy = DIRS[self.dir]
        straight = (hx + dx, hy + dy)
        right = (hx + DIRS[(self.dir+1) % 4][0], hy + DIRS[(self.dir+1) % 4][1])
        left  = (hx + DIRS[(self.dir-1) % 4][0], hy + DIRS[(self.dir-1) % 4][1])
        fx, fy = self.food
        return np.array([self._danger(straight), self._danger(right), self._danger(left),
                         self.dir == 2, self.dir == 0, self.dir == 3, self.dir == 1,
                         fx < hx, fx > hx, fy < hy, fy > hy], dtype=np.float32)

    def _food(self):
        while True:
            self.food = (self.rng.integers(self.grid), self.rng.integers(self.grid))
            if self.food not in self.snake: return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.rng = np.random.default_rng(seed)
        c = self.grid // 2
        self.snake = [(c, c), (c-1, c), (c-2, c)]; self.dir = 0
        self.score = 0; self.steps = 0; self._food()
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        if action == 1: self.dir = (self.dir+1) % 4
        elif action == 2: self.dir = (self.dir-1) % 4
        dx, dy = DIRS[self.dir]; head = (self.snake[0][0]+dx, self.snake[0][1]+dy)

        # --- cas 1 : le serpent MEURT (mur ou lui-même) -> épisode terminé ---
        if self._danger(head):
            reward = -10.0   # mourir coûte cher : il apprend à éviter murs et corps
            return self._obs(), reward, True, False, {"score": self.score}

        self.snake.insert(0, head)

        # --- cas 2 : il MANGE la pomme ---
        if head == self.food:
            self.score += 1
            reward = 10.0    # l'objectif : manger la pomme
            self._food()
        # --- cas 3 : déplacement NORMAL ---
        else:
            self.snake.pop()
            reward = -0.01   # petite pénalité par pas : évite de tourner en rond

        return self._obs(), reward, False, self.steps > 100*len(self.snake), {"score": self.score}


def regarder_jouer(model, n_parties=6):
    """Ouvre une fenêtre pygame et montre l'agent entraîné jouer."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((GRID*CELL, GRID*CELL + 40))
    pygame.display.set_caption("Snake — l'IA joue toute seule")
    font = pygame.font.SysFont("consolas", 22); clock = pygame.time.Clock()
    env = SnakeEnv()
    for partie in range(n_parties):
        obs, _ = env.reset(); done = False
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); return
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a)); done = term or trunc
            screen.fill((18, 24, 38))
            fx, fy = env.food
            pygame.draw.rect(screen, (230, 70, 70), (fx*CELL, fy*CELL, CELL, CELL))
            for i, (x, y) in enumerate(env.snake):
                col = (90, 200, 130) if i else (30, 150, 90)
                pygame.draw.rect(screen, col, (x*CELL+1, y*CELL+1, CELL-2, CELL-2))
            txt = font.render(f"Partie {partie+1}/{n_parties}   Score : {info['score']}", True, (230, 230, 230))
            screen.blit(txt, (8, GRID*CELL + 8)); pygame.display.flip(); clock.tick(12)
    pygame.quit()


def moyenne_pommes(politique, n=50):
    """Score moyen (pommes) d'une politique obs -> action sur n parties."""
    env, scores = SnakeEnv(), []
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i); done = False
        while not done:
            obs, r, term, trunc, info = env.step(politique(obs)); done = term or trunc
        scores.append(info["score"])
    return np.mean(scores), np.max(scores)


if __name__ == "__main__":
    import os
    from stable_baselines3 import DQN      # (vous pouvez aussi importer et tester PPO)
    from stable_baselines3.common.monitor import Monitor

    if os.path.exists("snake_dqn.zip"):    # déjà entraîné -> démo directe (supprimez le .zip pour ré-entraîner)
        model = DQN.load("snake_dqn")
        print(f"Agent DQN : {moyenne_pommes(lambda o: int(model.predict(o, deterministic=True)[0]))} (moyenne, max)")
        regarder_jouer(model); raise SystemExit

    env = Monitor(SnakeEnv(), "snake", info_keywords=("score",))   # log de chaque partie -> snake.monitor.csv
    print(f"Entraînement ({TIMESTEPS} pas)… patientez.")

    # DQN : actions discrètes + petit vecteur d'état. Variantes comparées : voir train.py.
    model = DQN("MlpPolicy", env, verbose=0,
                learning_rate=1e-3,
                buffer_size=100_000, learning_starts=1000, batch_size=64,
                gamma=0.9,                     # horizon court : suffisant avec ces capteurs locaux
                exploration_fraction=0.3, exploration_final_eps=0.05,
                target_update_interval=500, train_freq=4, seed=0)
    model.learn(total_timesteps=TIMESTEPS)
    model.save("snake_dqn")

    # Courbe d'apprentissage : pommes par partie (moyenne glissante sur 100 parties)
    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv("snake.monitor.csv", skiprows=1)
    plt.plot(df["l"].cumsum(), df["score"].rolling(100, min_periods=1).mean(), label="DQN")
    plt.axhline(0.1, color="grey", ls="--", label="hasard")
    plt.xlabel("pas d'entraînement"); plt.ylabel("pommes par partie"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Courbe d'apprentissage — Snake DQN"); plt.savefig("courbe.png", dpi=120)

    # Comparaison au hasard
    rng = np.random.default_rng(0)
    print(f"Hasard    : {moyenne_pommes(lambda o: int(rng.integers(3)))} (moyenne, max)")
    print(f"Agent DQN : {moyenne_pommes(lambda o: int(model.predict(o, deterministic=True)[0]))} (moyenne, max)")
    print("Entraînement terminé — courbe.png enregistrée — ouverture de la fenêtre.")
    regarder_jouer(model)
