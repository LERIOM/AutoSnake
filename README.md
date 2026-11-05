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
- **Algo** : DQN (actions discrètes + petit vecteur d'état), lr 1e-3, **γ 0.99**, ε 1 → 0.05 sur 30 % de l'entraînement, 150k pas.

## Résultats
Agent DQN (17 capteurs, γ 0.99) : **51,3 pommes en moyenne** sur 3 entraînements (49,5 / 51,2 / 53,3 ; max 76), évalués sur 50 parties chacun, contre **0,1** pour le hasard. Le modèle fourni (`snake_dqn.zip`) fait 51,8.

![courbe](courbe.png)

La courbe d'entraînement (~15 pommes) est plus basse que l'évaluation, parce que pendant l'entraînement l'agent joue encore 5 % de coups au hasard (ε = 0,05). Un seul mauvais coup suffit à tuer un long serpent.

### Ce qu'on a testé : progression
| étape | pommes |
|---|---|
| hasard | 0,1 |
| 11 capteurs fournis, γ 0.9 | ~22 |
| + 6 capteurs (danger à 2 cases, espace libre) | 37,4 |
| + γ 0.99 | **51,3** |

**1. L'état.** Avec les 11 capteurs fournis, l'agent plafonne vers 22 pommes et meurt en s'enroulant sur lui-même : il ne voit qu'une case autour de sa tête. Avec les 6 capteurs ajoutés, il atteint **37,4 pommes, soit ×1,7**.

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
| **lr 1e-3, γ 0.9, batch 64, réseau 64×64** (config d'origine) | **21,8** (20,5 à 23,6) |
| les pires (γ = 0.99 ou lr 2e-3) | 16,5 à 18 |

→ Avec 11 capteurs, aucun réglage ne fait mieux que le bruit entre seeds, et γ = 0.99 dégrade même les résultats : l'agent ne voit rien à anticiper.

**4. γ une fois les capteurs ajoutés : l'effet s'inverse.** L'agent voyait désormais les pièges se former, mais s'y jetait quand même : avec γ = 0.9, une mort 20 coups plus loin ne pèse que 0,9²⁰ ≈ 12 %. Une fois que l'agent a de quoi anticiper, regarder loin devient payant (17 capteurs, 3 seeds) :

| config | pommes |
|---|---|
| γ 0.9 | 37,4 (36,4 à 38,2) |
| γ 0.95 | 43,6 (42,0 à 45,0) |
| **γ 0.99** (retenue) | **51,3** (49,5 à 53,3) |
| γ 0.9, 400k pas, ε final 0.01 | 42,1 |
| γ 0.99, 400k pas, ε final 0.01 | 49,8 |

→ Entraîner plus longtemps n'aide qu'avec γ 0.9. On garde donc 150k pas.

**5. Malus d'enfermement** (*reward shaping*) : −5 ou −10 quand le serpent entre dans une zone plus petite que son corps.

| config | sans malus | malus −5 | malus −10 |
|---|---|---|---|
| γ 0.9 | 37,4 | 40,6 | 39,7 |
| γ 0.99 | 51,3 | 50,2 | 50,7 |

→ Le malus aide un peu avec γ 0.9, mais devient inutile avec γ 0.99 : la mort future pèse déjà assez pour que l'agent apprenne seul à éviter les poches. Il n'a donc pas été retenu.

**6. Deuxième entraînement avec des départs « serpent long »** (*curriculum*) : on reprend le modèle γ 0.99 et on l'entraîne 100k pas de plus. La moitié des parties démarrent depuis des positions de l'agent avec un long corps. On compare à un témoin entraîné aussi longtemps avec des départs normaux (3 seeds) :

| 2ᵉ entraînement | avant | après |
|---|---|---|
| témoin (départs normaux) | 51,3 | 52,1 |
| départs à 25-50 pommes | 51,3 | 49,9 |
| départs 10 à 30 coups avant chacune de ses morts | 51,3 | 50,7 |

→ Aucun gain : le manque d'expérience avec un long corps n'était pas le problème. La limite restante, c'est l'observation. Les 17 capteurs ne montrent ni la forme du corps ni la position de la queue, donc certaines poches restent imprévisibles. Piste suivante : donner la grille entière en entrée, avec un CNN.
