"""
jeu_snake_squelette.py — SQUELETTE ÉTUDIANT (Snake)
====================================================
Le code tourne déjà (fenêtre + entraînement), MAIS tant que la récompense n'est
pas définie l'agent n'apprend rien : c'est justement l'exercice de la séance.

À COMPLÉTER (cherchez les  # TODO) :
  1) la RÉCOMPENSE dans step()  ............... le cœur du MDP
  2) l'ENTRAÎNEMENT (algo + hyperparamètres) ... en bas du fichier
  (BONUS) enrichir l'OBSERVATION dans _obs()  -> fait : 6 capteurs en plus (17 au total)

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
        self.observation_space = spaces.Box(0.0, 1.0, (17,), np.float32)   # 11 fournis + 6 (BONUS)

    def _danger(self, pt):
        x, y = pt
        return x < 0 or x >= self.grid or y < 0 or y >= self.grid or (x, y) in self.snake

    def _espace_libre(self, depart):
        """BONUS : part des cases libres atteignables depuis `depart` (flood fill), 0 si `depart` est mortel.
        Permet à l'agent de « voir » qu'un virage l'enferme dans son propre corps."""
        if self._danger(depart): return 0.0
        vus, a_voir = {depart}, [depart]
        while a_voir:
            x, y = a_voir.pop()
            for dx, dy in DIRS:
                p = (x + dx, y + dy)
                if p not in vus and not self._danger(p):
                    vus.add(p); a_voir.append(p)
        return len(vus) / (self.grid * self.grid - len(self.snake))

    def _obs(self):
        # 11 "capteurs" FOURNIS : danger devant/droite/gauche, direction, position de la pomme.
        hx, hy = self.snake[0]; dx, dy = DIRS[self.dir]
        straight = (hx + dx, hy + dy)
        right = (hx + DIRS[(self.dir+1) % 4][0], hy + DIRS[(self.dir+1) % 4][1])
        left  = (hx + DIRS[(self.dir-1) % 4][0], hy + DIRS[(self.dir-1) % 4][1])
        fx, fy = self.food
        fournis = [self._danger(straight), self._danger(right), self._danger(left),
                   self.dir == 2, self.dir == 0, self.dir == 3, self.dir == 1,
                   fx < hx, fx > hx, fy < hy, fy > hy]
        # BONUS (+6) : danger à 2 cases devant/droite/gauche + espace libre si on va devant/droite/gauche
        loin = [self._danger((2*x - hx, 2*y - hy)) for x, y in (straight, right, left)]
        libre = [self._espace_libre(p) for p in (straight, right, left)]
        return np.array(fournis + loin + libre, dtype=np.float32)

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


HUD, PAD = 74, 18            # hauteur du bandeau de score, marge autour de la grille
FPS, VITESSE = 60, 10        # images par seconde, pas de jeu par seconde (↑/↓ pour changer)


def _mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class Rendu:
    """Rendu pygame animé : déplacement fluide entre les cases, anti-crénelage,
    particules quand il mange, secousse à la mort, écran de fin de partie."""
    S = 2                                       # sur-échantillonnage : on dessine en 2x puis on réduit (bords lisses)
    TETE, QUEUE = (150, 255, 160), (25, 120, 150)
    VERT, GRIS, ROUGE = (120, 235, 160), (135, 145, 172), (240, 70, 80)

    def __init__(self, n_parties):
        import pygame; self.pg = pygame
        pygame.init()
        self.W, self.H = GRID*CELL + 2*PAD, GRID*CELL + HUD + PAD
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Snake — l'IA joue toute seule   (espace : pause · ↑/↓ : vitesse)")
        self.c = pygame.Surface((self.W*self.S, self.H*self.S))
        police = "avenirnext,helveticaneue,helvetica,arial"
        self.f = {k: pygame.font.SysFont(police, t*self.S, bold=b)
                  for k, (t, b) in dict(titre=(26, True), petit=(14, False), score=(38, True),
                                        gros=(30, True), moyen=(17, True)).items()}
        self.fond = self._fond()
        self.n_parties, self.vitesse, self.pause = n_parties, VITESSE, False
        self.temps, self.clock = 0.0, pygame.time.Clock()
        self.particules, self.textes = [], []
        self.secousse = self.flash = 0.0
        self.pomme, self.pomme_t = None, 0.0

    # ---------- utilitaires ----------
    def px(self, cell):
        """Centre d'une case, en pixels du canevas sur-échantillonné."""
        return ((PAD + cell[0]*CELL + CELL/2) * self.S, (HUD + cell[1]*CELL + CELL/2) * self.S)

    def texte(self, txt, police, couleur, **pos):
        surf = self.f[police].render(txt, True, couleur)
        self.c.blit(surf, surf.get_rect(**pos)); return surf.get_rect(**pos)

    def _fond(self):
        pg, S = self.pg, self.S
        f = pg.Surface((self.W*S, self.H*S))
        for y in range(self.H*S):               # dégradé vertical bleu nuit
            pg.draw.line(f, _mix((20, 24, 42), (8, 10, 18), y / (self.H*S)), (0, y), (self.W*S, y))
        ox, oy, L = PAD*S, HUD*S, GRID*CELL*S
        pg.draw.rect(f, (6, 8, 14), (ox - 6*S, oy - 4*S, L + 12*S, L + 12*S), border_radius=16*S)   # ombre
        pg.draw.rect(f, (44, 52, 80), (ox - 5*S, oy - 5*S, L + 10*S, L + 10*S), border_radius=14*S)  # cadre
        pg.draw.rect(f, (16, 20, 34), (ox - 2*S, oy - 2*S, L + 4*S, L + 4*S), border_radius=11*S)
        for x in range(GRID):                   # tuiles arrondies en damier
            for y in range(GRID):
                col = (27, 33, 54) if (x + y) % 2 else (24, 29, 48)
                pg.draw.rect(f, col, (ox + (x*CELL + 1)*S, oy + (y*CELL + 1)*S, (CELL - 2)*S, (CELL - 2)*S),
                             border_radius=6*S)
        return f

    # ---------- événements de jeu ----------
    def manger(self, cell):
        import random
        x, y = self.px(cell)
        for _ in range(18):
            a, v = random.uniform(0, 6.283), random.uniform(1.5, 5.5) * self.S
            col = random.choice([(255, 90, 90), (255, 190, 80), (255, 240, 170), self.VERT])
            self.particules.append([x, y, v*np.cos(a), v*np.sin(a), 1.0, col])
        self.textes.append([x, y - CELL*self.S*0.4, 1.0])

    def mourir(self):
        self.secousse, self.flash = 1.0, 1.0

    # ---------- dessin ----------
    def _serpent(self, chemin, mort):
        """Blocs carrés qui glissent le long du chemin (tête en premier) : contour, dégradé, reflet, puis la tête."""
        pg, S = self.pg, self.S
        pts = []                                # échantillons (x, y, fraction depuis la tête)
        longueurs = [np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(chemin, chemin[1:])]
        total, cumul, pas = max(sum(longueurs), 1e-6), 0.0, 1.5*S
        for (a, b), l in zip(zip(chemin, chemin[1:]), longueurs):
            n = max(int(l / pas), 1)
            for k in range(n):
                t = k / n
                pts.append((a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, (cumul + l*t) / total))
            cumul += l
        pts.append((*chemin[-1], 1.0))
        R = CELL * 0.40 * S                     # demi-côté d'un bloc
        bloc = lambda col, x, y, h, rb=3: pg.draw.rect(self.c, col, (x - h, y - h, 2*h, 2*h), border_radius=int(rb*S))
        for x, y, fr in reversed(pts):          # contour sombre
            bloc((8, 30, 34), x, y, R*(1 - 0.15*fr) + 2*S, 5)
        for x, y, fr in reversed(pts):          # corps en dégradé
            bloc(_mix(self.TETE, self.QUEUE, fr), x, y, R*(1 - 0.15*fr), 4)
        for x, y, fr in reversed(pts):          # reflet : bande claire en haut à gauche
            bloc(_mix(_mix(self.TETE, self.QUEUE, fr), (255, 255, 255), 0.25), x - R*0.4, y - R*0.4, R*(1 - 0.15*fr)*0.25, 1)
        # tête : bloc un peu plus grand
        (hx, hy), (nx, ny) = chemin[0], chemin[1] if len(chemin) > 1 else chemin[0]
        d = np.array([hx - nx, hy - ny]); d = d / (np.linalg.norm(d) or 1)
        p = np.array([-d[1], d[0]])
        bloc((8, 30, 34), hx, hy, R*1.08 + 2*S, 6)
        bloc(self.TETE, hx, hy, R*1.08, 5)
        bloc((205, 255, 215), hx - R*0.4, hy - R*0.4, R*0.3, 1)
        if not mort and self.temps % 1.8 < 0.3:  # langue
            base = np.array([hx, hy]) + d*R*1.1; bout = base + d*7*S
            pg.draw.line(self.c, self.ROUGE, base, bout, int(2*S))
            for s in (-1, 1): pg.draw.line(self.c, self.ROUGE, bout, bout + d*3*S + p*s*3*S, int(2*S))
        cligne = self.temps % 3.3 < 0.12
        for s in (-1, 1):                       # yeux carrés
            ex, ey = np.array([hx, hy]) + d*R*0.35 + p*s*R*0.5
            if mort:
                for q in (1, -1):
                    pg.draw.line(self.c, (20, 25, 35), (ex - (d[0] + p[0]*q)*3*S, ey - (d[1] + p[1]*q)*3*S),
                                 (ex + (d[0] + p[0]*q)*3*S, ey + (d[1] + p[1]*q)*3*S), int(2*S))
            elif cligne:
                bloc((20, 25, 35), ex, ey, R*0.3, 0)
            else:
                bloc((250, 250, 250), ex, ey, R*0.3, 1)
                bloc((15, 20, 30), ex + d[0]*R*0.1, ey + d[1]*R*0.1, R*0.15, 0)

    def _pomme(self, cell):
        pg, S = self.pg, self.S
        if cell != self.pomme: self.pomme, self.pomme_t = cell, self.temps
        t = min((self.temps - self.pomme_t) / 0.3, 1.0)
        echelle = 1 + 2.7*(t - 1)**3 + 1.7*(t - 1)**2     # apparition avec léger rebond
        x, y = self.px(cell); y += np.sin(self.temps*4) * 1.5*S
        r = CELL * 0.33 * S * echelle
        halo = pg.Surface((CELL*2*S, CELL*2*S), pg.SRCALPHA)
        for i in range(4):
            h = CELL*S*(0.42 + 0.06*i) * echelle
            pg.draw.rect(halo, (255, 70, 80, 7), (CELL*S - h, CELL*S - h, 2*h, 2*h), border_radius=int(8*S))
        self.c.blit(halo, (x - CELL*S, y - CELL*S))
        carre = lambda col, cx, cy, h, rb: pg.draw.rect(self.c, col, (cx - h, cy - h, 2*h, 2*h), border_radius=int(rb*S))
        pg.draw.rect(self.c, (10, 12, 22), (x - r*0.8, y + r*0.95, r*1.6, r*0.35), border_radius=int(2*S))  # ombre
        carre((150, 25, 40), x, y, r, 5)                       # pomme carrée : bord sombre, face, reflet
        carre((232, 55, 70), x - r*0.07, y - r*0.07, r*0.86, 4)
        carre((255, 170, 170), x - r*0.42, y - r*0.42, r*0.2, 1)
        pg.draw.rect(self.c, (110, 70, 35), (x - 1.2*S, y - r*1.35, 2.5*S, r*0.45))            # queue
        pg.draw.rect(self.c, (95, 210, 110), (x + 1.5*S, y - r*1.35, r*0.7, r*0.32), border_radius=int(1*S))  # feuille

    def _effets(self):
        pg, S = self.pg, self.S
        for pa in self.particules:              # particules : inertie + gravité, rétrécissent en mourant
            pa[0] += pa[2]; pa[1] += pa[3]; pa[2] *= 0.92; pa[3] = pa[3]*0.92 + 0.12*S; pa[4] -= 1.6/FPS
            h = 3.5*S*pa[4]
            if pa[4] > 0: pg.draw.rect(self.c, pa[5], (pa[0] - h, pa[1] - h, 2*h, 2*h))
        self.particules = [pa for pa in self.particules if pa[4] > 0]
        for tx in self.textes:                  # « +1 » qui monte et s'efface
            tx[1] -= 0.7*S; tx[2] -= 1.2/FPS
            s = self.f["moyen"].render("+1", True, (255, 235, 150)); s.set_alpha(int(255*max(tx[2], 0)))
            self.c.blit(s, s.get_rect(center=(tx[0], tx[1])))
        self.textes = [tx for tx in self.textes if tx[2] > 0]

    def _hud(self, env, partie, record):
        pg, S = self.pg, self.S
        r = self.texte("SNAKE", "titre", (245, 248, 255), topleft=(PAD*S, 12*S))
        self.texte(" IA", "titre", self.VERT, topleft=r.topright)
        self.texte(f"Partie {partie}/{self.n_parties}   ·   Record {record}   ·   ×{self.vitesse/VITESSE:.1f}",
                   "petit", self.GRIS, topleft=(PAD*S, 46*S))
        r = self.texte(str(env.score), "score", (255, 255, 255), topright=((self.W - PAD)*S, 6*S))
        x, y = r.left - 18*S, r.centery + 2*S   # petite icône pomme à gauche du score
        pg.draw.rect(self.c, (232, 55, 70), (x - 8*S, y - 8*S, 16*S, 16*S), border_radius=int(3*S))
        pg.draw.rect(self.c, (255, 170, 170), (x - 5*S, y - 5*S, 4*S, 4*S))
        pg.draw.ellipse(self.c, (95, 210, 110), (x, y - 13*S, 8*S, 4*S))
        # barre de progression : longueur du serpent / taille de la grille
        L, X, Y = 110*S, (self.W - PAD)*S - 110*S, 56*S
        pg.draw.rect(self.c, (40, 48, 74), (X, Y, L, 5*S), border_radius=3*S)
        pg.draw.rect(self.c, self.VERT, (X, Y, max(L*len(env.snake)/GRID**2, 5*S), 5*S), border_radius=3*S)

    def _fin(self, env, record, t):
        pg, S = self.pg, self.S
        voile = pg.Surface(self.c.get_size(), pg.SRCALPHA); voile.fill((6, 8, 16, int(170*t)))
        self.c.blit(voile, (0, 0))
        cx, cy = self.W*S//2, (HUD + GRID*CELL/2)*S + (1 - t)*30*S
        box = pg.Rect(0, 0, 250*S, 150*S); box.center = (cx, cy)
        pg.draw.rect(self.c, (22, 28, 46), box, border_radius=18*S)
        pg.draw.rect(self.c, self.VERT, box, int(2*S), border_radius=18*S)
        self.texte("FIN DE PARTIE", "petit", self.GRIS, center=(cx, box.top + 26*S))
        self.texte(f"{env.score} pommes", "gros", (255, 255, 255), center=(cx, cy))
        nouveau = env.score >= record and env.score > 0
        self.texte("Nouveau record !" if nouveau else f"Record : {record}", "moyen",
                   (255, 215, 110) if nouveau else self.GRIS, center=(cx, box.bottom - 28*S))

    def image(self, env, chemin, partie, record, mort=False, fin=None):
        """Compose une image complète et l'affiche (fin = progression 0→1 de l'écran de fin)."""
        pg = self.pg
        self.c.blit(self.fond, (0, 0))
        self._hud(env, partie, record)
        self._pomme(env.food)
        self._serpent(chemin, mort)
        self._effets()
        if self.flash > 0:                      # flash rouge à la mort
            v = pg.Surface(self.c.get_size(), pg.SRCALPHA); v.fill((255, 40, 50, int(90*self.flash)))
            self.c.blit(v, (0, 0)); self.flash -= 2.5/FPS
        if fin is not None: self._fin(env, record, fin)
        dx = dy = 0
        if self.secousse > 0:                   # tremblement de l'écran
            dx, dy = np.random.uniform(-1, 1, 2) * 6 * self.secousse; self.secousse -= 2/FPS
        self.screen.fill((8, 10, 18))
        self.screen.blit(pg.transform.smoothscale(self.c, (self.W, self.H)), (dx, dy))
        pg.display.flip()
        self.temps += 1/FPS

    def evenements(self):
        """Clavier/fenêtre. Renvoie False si on ferme la fenêtre."""
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE): return False
            if e.type == pg.KEYDOWN and e.key == pg.K_SPACE: self.pause = not self.pause
            if e.type == pg.KEYDOWN and e.key == pg.K_UP: self.vitesse = min(self.vitesse + 2, 40)
            if e.type == pg.KEYDOWN and e.key == pg.K_DOWN: self.vitesse = max(self.vitesse - 2, 2)
        self.clock.tick(FPS)
        return True


def chemin_anime(r, avant, apres, t):
    """Points du corps à l'instant t ∈ [0,1] entre deux pas : la tête avance, la queue suit,
    et les coudes restent au centre des cases -> virages bien arrondis."""
    p = lambda a, b: (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)
    A = [r.px(c) for c in avant]
    if apres == avant: return A                 # il vient de mourir : il ne bouge plus
    tete = p(A[0], r.px(apres[0]))
    if len(apres) > len(avant): return [tete] + A             # il a mangé : la queue reste en place
    return [tete] + A[:-1] + [p(A[-1], A[-2])]


def regarder_jouer(model, n_parties=6):
    """Ouvre une fenêtre pygame et montre l'agent entraîné jouer."""
    r, env, record = Rendu(n_parties), SnakeEnv(), 0
    for partie in range(1, n_parties + 1):
        obs, _ = env.reset(); done = False
        while not done:
            avant, pomme, score = list(env.snake), env.food, env.score
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(a)); done = term or trunc
            if env.score > score: r.manger(pomme)
            if term: r.mourir()
            record = max(record, env.score)
            n_img, k = max(1, round(FPS / r.vitesse)), 0
            while k < n_img:                    # images intermédiaires entre deux pas
                if not r.evenements(): r.pg.quit(); return
                if not r.pause: k += 1
                r.image(env, chemin_anime(r, avant, list(env.snake), k / n_img), partie, record, mort=term)
        for i in range(int(FPS * 2.2)):         # écran de fin (~2 s)
            if not r.evenements(): r.pg.quit(); return
            r.image(env, chemin_anime(r, env.snake, env.snake, 1), partie, record, mort=term,
                    fin=min(i / (FPS*0.4), 1.0))
    r.pg.quit()


def moyenne_pommes(politique, n=50):
    """Score moyen (pommes) d'une politique obs -> action sur n parties."""
    env, scores = SnakeEnv(), []
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i); done = False
        while not done:
            obs, r, term, trunc, info = env.step(politique(obs)); done = term or trunc
        scores.append(info["score"])
    return f"{np.mean(scores):.1f} pommes en moyenne (max {max(scores)})"


if __name__ == "__main__":
    import os
    from stable_baselines3 import DQN      # (vous pouvez aussi importer et tester PPO)
    from stable_baselines3.common.monitor import Monitor

    if os.path.exists("snake_dqn.zip"):    # déjà entraîné -> démo directe (supprimez le .zip pour ré-entraîner)
        model = DQN.load("snake_dqn")
        print(f"Agent DQN : {moyenne_pommes(lambda o: int(model.predict(o, deterministic=True)[0]))}")
        regarder_jouer(model); raise SystemExit

    env = Monitor(SnakeEnv(), "snake", info_keywords=("score",))   # log de chaque partie -> snake.monitor.csv
    print(f"Entraînement ({TIMESTEPS} pas)… patientez.")

    # DQN : actions discrètes + petit vecteur d'état. Variantes comparées : voir train.py.
    model = DQN("MlpPolicy", env, verbose=0,
                learning_rate=1e-3,
                buffer_size=100_000, learning_starts=1000, batch_size=64,
                gamma=0.99,                    # horizon long : il anticipe les pièges (0.9 -> 37 pommes, 0.99 -> 51)
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
    print(f"Hasard    : {moyenne_pommes(lambda o: int(rng.integers(3)))}")
    print(f"Agent DQN : {moyenne_pommes(lambda o: int(model.predict(o, deterministic=True)[0]))}")
    print("Entraînement terminé — courbe.png enregistrée — ouverture de la fenêtre.")
    regarder_jouer(model)
