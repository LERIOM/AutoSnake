"""
demo.py — regarder un modèle entraîné jouer (sans ré-entraîner).

  python demo.py                        # meilleur modèle par défaut (r3_plus)
  python demo.py r2_base --parties 3 --fps 15
  python demo.py r3_plus --gif demo.gif # enregistre une vidéo de secours (sans fenêtre)
"""
import argparse, json, os
p = argparse.ArgumentParser()
p.add_argument("model", nargs="?", default="r3_plus")
p.add_argument("--parties", type=int, default=6)
p.add_argument("--fps", type=int, default=12)
p.add_argument("--gif", default=None)
args = p.parse_args()
if args.gif: os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from stable_baselines3 import DQN
from snake_env import SnakeEnv, GRID, CELL

meta = json.load(open(f"models/{args.model}.json"))
model = DQN.load(f"models/{args.model}")
env = SnakeEnv(obs_mode=meta["obs"])

pygame.init()
screen = pygame.display.set_mode((GRID*CELL, GRID*CELL + 40))
pygame.display.set_caption(f"Snake — l'IA joue toute seule ({args.model})")
font = pygame.font.SysFont("consolas", 22); clock = pygame.time.Clock()
frames, scores = [], []

for partie in range(args.parties):
    obs, _ = env.reset(seed=partie); done = False
    while not done:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); raise SystemExit
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(a)); done = term or trunc
        screen.fill((18, 24, 38))
        fx, fy = env.food
        pygame.draw.rect(screen, (230, 70, 70), (fx*CELL, fy*CELL, CELL, CELL))
        for i, (x, y) in enumerate(env.snake):
            col = (90, 200, 130) if i else (30, 150, 90)
            pygame.draw.rect(screen, col, (x*CELL+1, y*CELL+1, CELL-2, CELL-2))
        txt = font.render(f"Partie {partie+1}/{args.parties}   Score : {info['score']}", True, (230, 230, 230))
        screen.blit(txt, (8, GRID*CELL + 8)); pygame.display.flip()
        if args.gif: frames.append(pygame.image.tobytes(screen, "RGB"))
        else: clock.tick(args.fps)
    scores.append(info["score"]); print(f"partie {partie+1} : {info['score']} pommes")
pygame.quit()

if args.gif:
    from PIL import Image
    size = (GRID*CELL, GRID*CELL + 40)
    imgs = [Image.frombytes("RGB", size, f) for f in frames]
    imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=1000 // args.fps, loop=0)
    print(f"-> {args.gif} ({len(imgs)} images)")
