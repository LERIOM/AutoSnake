"""
train.py — entraîne un DQN sur une variante du MDP et sauvegarde le modèle.

Exemples :
  python train.py --reward r2 --obs base                 -> models/r2_base.zip
  python train.py --reward r3 --obs plus --steps 200000  -> models/r3_plus.zip
  python train.py --reward r2 --gamma 0.99 --name r2_base_g099

Chaque épisode est loggé dans logs/<nom>.monitor.csv (récompense, durée, score) -> plot.py.
"""
import argparse, json, os, time
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv

p = argparse.ArgumentParser()
p.add_argument("--reward", default="r2", choices=["r1", "r2", "r3"])
p.add_argument("--obs", default="base", choices=["base", "plus"])
p.add_argument("--steps", type=int, default=150_000)
p.add_argument("--gamma", type=float, default=0.9)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--name", default=None)
args = p.parse_args()
name = args.name or f"{args.reward}_{args.obs}"

torch.set_num_threads(1)              # plusieurs entraînements peuvent tourner en parallèle
os.makedirs("logs", exist_ok=True); os.makedirs("models", exist_ok=True)
env = Monitor(SnakeEnv(reward_mode=args.reward, obs_mode=args.obs),
              f"logs/{name}", info_keywords=("score",))

model = DQN("MlpPolicy", env, verbose=0,
            learning_rate=args.lr,
            buffer_size=100_000, learning_starts=1000, batch_size=64,
            gamma=args.gamma,
            exploration_fraction=0.3, exploration_final_eps=0.05,
            target_update_interval=500, train_freq=4, seed=args.seed)

print(f"[{name}] entraînement {args.steps} pas…", flush=True)
t0 = time.time()
model.learn(total_timesteps=args.steps)
model.save(f"models/{name}")
with open(f"models/{name}.json", "w") as f:
    json.dump(vars(args) | {"name": name, "train_seconds": round(time.time() - t0)}, f, indent=2)
print(f"[{name}] terminé en {time.time() - t0:.0f}s -> models/{name}.zip", flush=True)
