"""
snake_env.py — environnement Gymnasium du Snake, avec variantes de MDP
=======================================================================
Reprend l'env du squelette et paramètre les deux choix qu'on compare :

  reward_mode :
    "r1"  +10 pomme · −10 mort · 0 par pas
    "r2"  +10 pomme · −10 mort · −0,01 par pas            (proposition de la fiche)
    "r3"  r2 + shaping distance : +0,1 si on se rapproche de la pomme, −0,1 sinon

  obs_mode :
    "base"  11 capteurs 0/1 (danger devant/droite/gauche, direction, position pomme)
    "plus"  base + 6 capteurs (bonus) :
              - danger à 2 cases devant/droite/gauche
              - part des cases libres atteignables (flood fill) en allant devant/droite/gauche
            -> l'agent « voit » quand un virage l'enferme dans son propre corps.
"""
import warnings; warnings.filterwarnings("ignore")
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

GRID = 12
CELL = 32
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # droite, bas, gauche, haut

REWARDS = {"r1": dict(eat=10.0, die=-10.0, step=0.0,   shaping=0.0),
           "r2": dict(eat=10.0, die=-10.0, step=-0.01, shaping=0.0),
           "r3": dict(eat=10.0, die=-10.0, step=-0.01, shaping=0.1)}
OBS_SIZE = {"base": 11, "plus": 17}


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid=GRID, reward_mode="r2", obs_mode="base"):
        super().__init__()
        self.grid = grid
        self.reward_mode, self.obs_mode = reward_mode, obs_mode
        self.R = REWARDS[reward_mode]
        self.action_space = spaces.Discrete(3)                   # 0=tout droit, 1=tourner à droite, 2=à gauche
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_SIZE[obs_mode],), np.float32)

    def _danger(self, pt):
        x, y = pt
        return x < 0 or x >= self.grid or y < 0 or y >= self.grid or (x, y) in self.snake

    def _free_space(self, start):
        """Part des cases libres atteignables depuis `start` (0 si `start` est mortelle)."""
        if self._danger(start): return 0.0
        body = set(self.snake)
        seen, todo = {start}, deque([start])
        while todo:
            x, y = todo.popleft()
            for dx, dy in DIRS:
                p = (x + dx, y + dy)
                if p not in seen and 0 <= p[0] < self.grid and 0 <= p[1] < self.grid and p not in body:
                    seen.add(p); todo.append(p)
        return len(seen) / (self.grid * self.grid - len(self.snake))

    def _obs(self):
        hx, hy = self.snake[0]
        d_s, d_r, d_l = DIRS[self.dir], DIRS[(self.dir+1) % 4], DIRS[(self.dir-1) % 4]
        straight, right, left = [(hx + dx, hy + dy) for dx, dy in (d_s, d_r, d_l)]
        fx, fy = self.food
        obs = [self._danger(straight), self._danger(right), self._danger(left),
               self.dir == 2, self.dir == 0, self.dir == 3, self.dir == 1,
               fx < hx, fx > hx, fy < hy, fy > hy]
        if self.obs_mode == "plus":
            obs += [self._danger((hx + 2*dx, hy + 2*dy)) for dx, dy in (d_s, d_r, d_l)]
            obs += [self._free_space(p) for p in (straight, right, left)]
        return np.array(obs, dtype=np.float32)

    def _food(self):
        while True:
            self.food = (int(self.rng.integers(self.grid)), int(self.rng.integers(self.grid)))
            if self.food not in self.snake: return

    def _dist(self):
        return abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])

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
        old_dist = self._dist()

        # --- cas 1 : le serpent MEURT (mur ou lui-même) -> épisode terminé ---
        if self._danger(head):
            return self._obs(), self.R["die"], True, False, {"score": self.score}

        self.snake.insert(0, head)

        # --- cas 2 : il MANGE la pomme ---
        if head == self.food:
            self.score += 1
            reward = self.R["eat"]
            self._food()
        # --- cas 3 : déplacement NORMAL ---
        else:
            self.snake.pop()
            reward = self.R["step"]
            if self.R["shaping"]:
                reward += self.R["shaping"] if self._dist() < old_dist else -self.R["shaping"]

        return self._obs(), reward, False, self.steps > 100*len(self.snake), {"score": self.score}
