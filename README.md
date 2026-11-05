# Projet RL — Snake (DQN, Stable-Baselines3)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python jeu_snake_squelette.py
```
- 1er lancement : entraînement (~1 min) → `snake_dqn.zip`, `courbe.png`, score comparé au hasard, puis démo.
- Lancements suivants : charge `snake_dqn.zip` et lance directement la démo. Supprimez ce fichier pour ré-entraîner.

## Le MDP
- **État** (17 variables entre 0 et 1) :
  - 11 fournies : danger devant, à droite et à gauche · direction actuelle (4) · pomme à gauche, à droite, en haut, en bas ;
  - 6 ajoutées (bonus) : danger **à 2 cases** devant, à droite et à gauche · **part des cases libres encore atteignables** (*flood fill*) si l'on va devant, à droite ou à gauche. Avec ces capteurs, l'agent voit qu'un virage l'enfermerait dans son propre corps.
- **Actions** (3, relatives) : tout droit, tourner à droite, tourner à gauche. Le demi-tour est impossible par construction.
- **Récompense** : +10 quand il mange une pomme · −10 quand il meurt · −0,01 par pas, pour qu'il ne tourne pas en rond.
- **Fin d'épisode** : à la mort, ou après 100 × longueur pas (anti-boucle infinie).
- **Algo** : DQN (actions discrètes + petit vecteur d'état), lr 1e-3, γ 0.9, ε 1 → 0.05 sur 30 % de l'entraînement, 150k pas.

## Résultats
Agent DQN (17 capteurs) : **37,4 pommes en moyenne** sur 3 entraînements (37,6 / 38,2 / 36,4 ; max 63), évalués sur 50 parties chacun, contre **0,1** pour le hasard.

![courbe](courbe.png)

La courbe d'entraînement (~15 pommes) est plus basse que l'évaluation, parce que pendant l'entraînement l'agent joue encore 5 % de coups au hasard (ε = 0,05). Un seul mauvais coup suffit à tuer un long serpent.

### Ce qu'on a testé
**1. L'état : c'est ce qui compte.** Avec les 11 capteurs fournis, l'agent plafonne vers 20 pommes et meurt en s'enroulant sur lui-même : il ne voit qu'une case autour de sa tête. Avec les 6 capteurs ajoutés, il atteint **37,4 pommes, soit ×1,7**.

**2. La récompense** (11 capteurs, 100 parties) :

| variante | pommes |
|---|---|
| +10 / −10 / 0 (sans pénalité par pas) | 18,0 |
| **+10 / −10 / −0,01** (retenue) | **22,9** |
| + bonus ±0,1 si l'agent se rapproche ou s'éloigne de la pomme | 19,1 |

**3. Les hyperparamètres** : recherche aléatoire sur 28 configurations (lr, γ, batch, exploration, mise à jour du réseau cible, fréquence d'entraînement, taille du réseau), puis revalidation des 5 meilleures sur 3 seeds.

| config | pommes (moyenne sur 3 seeds) |
|---|---|
| lr 5e-4, batch 128, réseau 256×256, `train_freq=1` | 22,8 (19,0 à 25,8) |
| **lr 1e-3, γ 0.9, batch 64, réseau 64×64** (retenue) | **21,8** (20,5 à 23,6) |
| les pires (γ = 0.99 ou lr 2e-3) | 16,5 à 18 |

→ Aucun réglage ne fait mieux que le bruit entre seeds. On garde la config d'origine, 2× plus rapide à entraîner. γ = 0.99 dégrade les résultats : avec des capteurs aussi locaux, regarder loin dans le futur n'aide pas.
