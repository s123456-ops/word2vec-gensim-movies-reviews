from pathlib import Path

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "Results"
FIGURES_DIR = PROJECT_DIR / "Report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def acp_2d(X):
    """ACP (analyse en composantes principales) en 2 dimensions, avec numpy seulement.
    Remplace sklearn.decomposition.PCA : meme calcul, pas besoin d'installer scikit-learn.
    Renvoie les coordonnees 2D et la part de variance expliquee par chaque axe."""
    Xc = X - X.mean(axis=0)                      # 1) on centre les donnees
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # 2) decomposition en valeurs singulieres
    # 3) meme convention de signe que scikit-learn (evite une figure inversee)
    signes = np.sign(Vt[range(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
    U = U * signes
    Y = U[:, :2] * S[:2]                         # 4) coordonnees sur les 2 premiers axes
    variance = S**2 / np.sum(S**2)               # 5) part de variance de chaque axe
    return Y, variance[:2]


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
cfg = {"films": (RESULTS_DIR / "movie_word2vec_clean.model", ["love", "war", "family", "death", "man"]),
       "avis": (RESULTS_DIR / "reviews_word2vec_corrected.model", ["phone", "battery", "screen", "quality", "good"])}
cols = ["#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910"]

for name, (path, qs) in cfg.items():
    wv = Word2Vec.load(str(path)).wv
    print(name, len(wv), {q: [(w, round(float(s), 4)) for w, s in wv.most_similar(q, topn=5)] for q in qs})
    words, groups = [], []
    for i, q in enumerate(qs):
        words.append(q); groups.append((i, True))
        for w, _ in wv.most_similar(q, topn=5):
            words.append(w); groups.append((i, False))
    X = np.array([wv.get_vector(w, norm=True) for w in words])
    Y, ev = acp_2d(X)
    print(name, "variance", ev, ev.sum())
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for (i, isq), (x, y), w in zip(groups, Y, words):
        ax.scatter(x, y, s=60 if isq else 22, color=cols[i], marker="s" if isq else "o", zorder=3, edgecolor="white", linewidth=.6)
        ax.text(x, y, w, fontsize=9.5 if isq else 8, fontweight="bold" if isq else "normal", color=cols[i], zorder=4)
    ax.set_xlabel(f"Composante principale 1 ({ev[0]*100:.1f} % de la variance)")
    ax.set_ylabel(f"Composante principale 2 ({ev[1]*100:.1f} %)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(alpha=.25, linewidth=.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"acp_{name}.pdf")
    fig.savefig(FIGURES_DIR / f"acp_{name}.png", dpi=150)
    plt.close(fig)
    print("Figure enregistree :", FIGURES_DIR / f"acp_{name}.png")