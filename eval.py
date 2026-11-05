"""
eval.py — compare chaque modèle de models/ à un agent qui joue au hasard.

  python eval.py                 # tous les modèles, 100 parties chacun
  python eval.py --n 200 r3_plus # un seul modèle

Mêmes graines de parties pour tout le monde (comparaison équitable).
Résultats affichés + écrits dans results.md.
"""
import argparse, glob, json, os
import numpy as np
from stable_baselines3 import DQN
from snake_env import SnakeEnv

p = argparse.ArgumentParser()
p.add_argument("models", nargs="*")
p.add_argument("--n", type=int, default=100)
args = p.parse_args()


def play(policy, obs_mode, n):
    env = SnakeEnv(obs_mode=obs_mode)
    scores = []
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i); done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs)); done = term or trunc
        scores.append(info["score"])
    return np.array(scores)


rng = np.random.default_rng(0)
rows = [("hasard", "-", "-", play(lambda o: int(rng.integers(3)), "base", args.n))]
names = args.models or sorted(os.path.basename(f)[:-4] for f in glob.glob("models/*.zip"))
for name in names:
    meta = json.load(open(f"models/{name}.json"))
    model = DQN.load(f"models/{name}")
    s = play(lambda o: int(model.predict(o, deterministic=True)[0]), meta["obs"], args.n)
    rows.append((name, meta["reward"], f"{meta['obs']} γ={meta['gamma']}", s))

lines = [f"Évaluation sur {args.n} parties (grille 12×12, politique déterministe)\n",
         "| modèle | récompense | obs / γ | moyenne | médiane | max | ≥10 pommes |",
         "|---|---|---|---|---|---|---|"]
for name, rew, cfg, s in rows:
    lines.append(f"| {name} | {rew} | {cfg} | {s.mean():.1f} ± {s.std():.1f} | {np.median(s):.0f} "
                 f"| {s.max()} | {(s >= 10).mean():.0%} |")
print("\n".join(lines))
open("results.md", "w").write("\n".join(lines) + "\n")
