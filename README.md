# Projet RL — Snake (DQN, Stable-Baselines3)

## Installation
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Fichiers
| fichier | rôle |
|---|---|
| `jeu_snake_squelette.py` | squelette du cours, TODO complétés (version R2, tout-en-un) |
| `snake_env.py` | env Gymnasium paramétrable : `reward_mode` (r1/r2/r3), `obs_mode` (base/plus) |
| `train.py` | entraîne un DQN → `models/<nom>.zip` + `logs/<nom>.monitor.csv` |
| `eval.py` | compare chaque modèle au hasard sur 100 parties → `results.md` |
| `plot.py` | courbes d'apprentissage → `courbes.png` |
| `demo.py` | regarder jouer un modèle sauvegardé (`--gif` pour une vidéo de secours) |

## Commandes
```bash
.venv/bin/python train.py --reward r3 --obs plus      # ~3-5 min
.venv/bin/python eval.py
.venv/bin/python plot.py
.venv/bin/python demo.py r3_plus
.venv/bin/python demo.py r3_plus --gif demo.gif --parties 2
```

## Le MDP
- **État** : 11 capteurs 0/1 (version `base`) : danger devant, à droite et à gauche · direction (4) · pomme à gauche, à droite, en haut, en bas.
  La version `plus` ajoute 6 capteurs (17 en tout) : danger à 2 cases devant, à droite et à gauche, et **part des cases libres atteignables** (flood fill) si l'on va devant, à droite ou à gauche. Avec ces capteurs, l'agent voit qu'un virage l'enferme dans son corps.
- **Actions** : 3 actions relatives : tout droit, tourner à droite, tourner à gauche. Le demi-tour (suicide immédiat) est impossible par construction.
- **Récompense** (variantes comparées) :

| variante | pomme | mort | pas normal | shaping |
|---|---|---|---|---|
| R1 | +10 | −10 | 0 | — |
| R2 | +10 | −10 | −0,01 | — |
| R3 | +10 | −10 | −0,01 | +0,1 si on se rapproche de la pomme, −0,1 sinon |

- **Fin d'épisode** : mort (épisode *terminé*), ou plus de 100 × longueur pas joués (*tronqué*), pour éviter les boucles infinies.
- **Algorithme** : DQN, adapté à des actions discrètes et à un petit vecteur d'état. MLP par défaut de SB3 (64-64), lr 1e-3, γ 0.9, buffer 100k, ε : 1 → 0.05 sur 30 % de l'entraînement, réseau cible mis à jour tous les 500 pas.

## Résultats
Voir `results.md` (tableau) et `courbes.png` (courbes d'apprentissage).
