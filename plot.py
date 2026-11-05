"""
plot.py — courbes d'apprentissage à partir de logs/*.monitor.csv -> courbes.png

  python plot.py                  # toutes les variantes
  python plot.py r2_base r3_plus  # seulement celles-ci
"""
import glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

names = sys.argv[1:] or sorted(os.path.basename(f).split(".")[0] for f in glob.glob("logs/*.monitor.csv"))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name in names:
    df = pd.read_csv(f"logs/{name}.monitor.csv", skiprows=1)
    x = df["l"].cumsum()                                  # pas d'entraînement cumulés
    axes[0].plot(x, df["score"].rolling(100, min_periods=1).mean(), label=name)
    axes[1].plot(x, df["r"].rolling(100, min_periods=1).mean(), label=name)

axes[0].axhline(0.2, color="grey", ls="--", label="hasard (~0,2)")
axes[0].set(title="Pommes mangées par partie (moyenne glissante 100 parties)",
            xlabel="pas d'entraînement", ylabel="pommes")
axes[1].set(title="Récompense cumulée par partie (moyenne glissante 100)",
            xlabel="pas d'entraînement", ylabel="récompense")
axes[1].text(0.98, 0.02, "échelles différentes selon la récompense :\ncomparer les pommes à gauche",
             transform=axes[1].transAxes, fontsize=8, color="grey", ha="right")
fig.text(0.5, 0.005, "pendant l'entraînement l'agent explore encore (ε = 0,05) : "
         "les scores d'évaluation (eval.py, sans exploration) sont plus élevés", ha="center", fontsize=8, color="grey")
for ax in axes: ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig("courbes.png", dpi=130)
print("-> courbes.png")
