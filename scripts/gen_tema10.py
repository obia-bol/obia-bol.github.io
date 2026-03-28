"""Genera 9 graficos para tema-10: Introduccion a Redes Neuronales."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import warnings

warnings.filterwarnings("ignore")

OUT = "public/ruta-aprendizaje-graficos/tema-10"
np.random.seed(42)

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"
CYAN = "#64B5CD"
GRAY = "#8c8c8c"
BG = "#f8f9fa"


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  saved {name}")


# ── 01 Funciones de activacion ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=BG)
fig.suptitle("Funciones de Activación", fontsize=14, fontweight="bold")
x = np.linspace(-4, 4, 400)

acts = {
    "Sigmoid": (lambda x: 1 / (1 + np.exp(-x)), BLUE),
    "Tanh": (lambda x: np.tanh(x), ORANGE),
    "ReLU": (lambda x: np.maximum(0, x), GREEN),
    "Leaky ReLU α=0.1": (lambda x: np.where(x > 0, x, 0.1 * x), RED),
    "ELU α=1": (lambda x: np.where(x > 0, x, np.exp(x) - 1), PURPLE),
    "Softplus": (lambda x: np.log1p(np.exp(x)), CYAN),
}

for ax, (name, (fn, color)) in zip(axes.flat, acts.items()):
    ax.set_facecolor(BG)
    y = fn(x)
    ax.plot(x, y, color=color, linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("z")
    ax.set_ylabel("σ(z)")
    ax.grid(True, alpha=0.3)
    # anotar derivada en 0
    dy0 = float(np.gradient(fn(x), x)[np.argmin(np.abs(x))])
    ax.annotate(
        f"σ'(0)≈{dy0:.2f}",
        xy=(0, fn(np.array([0]))[0]),
        xytext=(1.2, fn(np.array([0]))[0] - 0.4),
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1),
    )

savefig("01-funciones-activacion.png")

# ── 02 Neurona / Perceptron ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Anatomía de una Neurona Artificial", fontsize=13, fontweight="bold")

# entradas
inputs = [
    (0.8, 6.5, "x₁", "w₁=0.8"),
    (0.8, 4.0, "x₂", "w₂=-0.3"),
    (0.8, 1.5, "x₃", "w₃=1.2"),
]
for xp, yp, xlabel, wlabel in inputs:
    ax.add_patch(plt.Circle((xp, yp), 0.4, color=CYAN, zorder=5))
    ax.text(xp, yp, xlabel, ha="center", va="center", fontsize=11, fontweight="bold")
    # flecha al soma
    ax.annotate(
        "",
        xy=(4.2, 4.0),
        xytext=(xp + 0.4, yp),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=GRAY),
    )
    ax.text(
        (xp + 0.4 + 4.2) / 2,
        (yp + 4.0) / 2 + 0.15,
        wlabel,
        fontsize=9,
        color=RED,
        ha="center",
    )

# bias
ax.add_patch(plt.Circle((2.5, 7.0), 0.4, color=ORANGE, zorder=5))
ax.text(2.5, 7.0, "b", ha="center", va="center", fontsize=11, fontweight="bold")
ax.annotate(
    "",
    xy=(4.2, 4.4),
    xytext=(2.9, 7.0),
    arrowprops=dict(arrowstyle="->", lw=1.5, color=ORANGE),
)

# soma
ax.add_patch(plt.Circle((4.8, 4.0), 0.8, color=BLUE, zorder=5, alpha=0.9))
ax.text(
    4.8,
    4.0,
    "∑",
    ha="center",
    va="center",
    fontsize=16,
    color="white",
    fontweight="bold",
)

# activacion
ax.add_patch(plt.Circle((7.0, 4.0), 0.8, color=PURPLE, zorder=5, alpha=0.9))
ax.text(
    7.0,
    4.0,
    "σ",
    ha="center",
    va="center",
    fontsize=16,
    color="white",
    fontweight="bold",
)

ax.annotate(
    "",
    xy=(6.2, 4.0),
    xytext=(5.6, 4.0),
    arrowprops=dict(arrowstyle="->", lw=2, color=GRAY),
)
ax.text(5.9, 4.3, "z = Σwᵢxᵢ+b", fontsize=9, ha="center")

# salida
ax.add_patch(plt.Circle((9.2, 4.0), 0.4, color=GREEN, zorder=5))
ax.text(9.2, 4.0, "ŷ", ha="center", va="center", fontsize=12, fontweight="bold")
ax.annotate(
    "",
    xy=(8.8, 4.0),
    xytext=(7.8, 4.0),
    arrowprops=dict(arrowstyle="->", lw=2, color=GRAY),
)
ax.text(8.3, 4.3, "a=σ(z)", fontsize=9, ha="center")

# etiquetas capas
for xp, label in [(4.8, "Soma\n(suma ponderada)"), (7.0, "Activación\n(no lineal)")]:
    ax.text(xp, 2.8, label, ha="center", fontsize=9, color=GRAY)

savefig("02-neurona-perceptron.png")

# ── 03 Red multicapa - forward pass ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Red Multicapa: Forward Pass", fontsize=13, fontweight="bold")

layer_x = [1.0, 3.5, 6.0, 8.5]
layer_n = [3, 4, 4, 2]
layer_labels = [
    "Entrada\n(x)",
    "Capa oculta 1\nReLU",
    "Capa oculta 2\nReLU",
    "Salida\nSoftmax",
]
layer_colors = [CYAN, BLUE, PURPLE, GREEN]
node_pos = {}

for li, (lx, n, lbl, lc) in enumerate(
    zip(layer_x, layer_n, layer_labels, layer_colors)
):
    ys = np.linspace(7 - (n - 1) * 1.2, 7, n)
    for ni, yp in enumerate(ys):
        node_pos[(li, ni)] = (lx, yp)
        ax.add_patch(plt.Circle((lx, yp), 0.38, color=lc, zorder=5, alpha=0.85))
    # conexiones
    if li > 0:
        prev_n = layer_n[li - 1]
        for pi in range(prev_n):
            for ci in range(n):
                px, py = node_pos[(li - 1, pi)]
                cx, cy = node_pos[(li, ci)]
                ax.plot(
                    [px + 0.38, cx - 0.38],
                    [py, cy],
                    color=GRAY,
                    alpha=0.25,
                    lw=0.8,
                    zorder=1,
                )
    ax.text(lx, 1.5, lbl, ha="center", fontsize=9, color=lc, fontweight="bold")

# formula debajo
ax.text(
    5.0,
    0.5,
    "ŷ = softmax(W₂·ReLU(W₁·ReLU(W₀x + b₀) + b₁) + b₂)",
    ha="center",
    fontsize=10,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRAY, alpha=0.8),
)

savefig("03-forward-pass-red.png")

# ── 04 Backpropagation - flujo de gradientes ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle(
    "Backpropagation: Gradientes y Descenso del Gradiente",
    fontsize=13,
    fontweight="bold",
)

# izq: superficie de perdida y trayectoria GD
ax = axes[0]
ax.set_facecolor(BG)
w1 = np.linspace(-3, 3, 200)
w2 = np.linspace(-3, 3, 200)
W1, W2 = np.meshgrid(w1, w2)
Z = W1**2 + 3 * W2**2 + 0.5 * W1 * W2
c = ax.contourf(W1, W2, Z, levels=20, cmap="Blues", alpha=0.7)
plt.colorbar(c, ax=ax, label="Loss")
# trayectoria GD
wpath = np.array([[2.5, 2.0]])
lr = 0.15
for _ in range(30):
    g1 = 2 * wpath[-1, 0] + 0.5 * wpath[-1, 1]
    g2 = 6 * wpath[-1, 1] + 0.5 * wpath[-1, 0]
    new_w = wpath[-1] - lr * np.array([g1, g2])
    wpath = np.vstack([wpath, new_w])
ax.plot(
    wpath[:, 0],
    wpath[:, 1],
    "o-",
    color=RED,
    markersize=4,
    linewidth=1.5,
    label="GD path",
)
ax.plot(*wpath[0], "s", color=ORANGE, markersize=8, label="inicio")
ax.plot(*wpath[-1], "*", color=GREEN, markersize=12, label="mínimo")
ax.set_xlabel("w₁")
ax.set_ylabel("w₂")
ax.set_title("Superficie de Loss + Trayectoria GD")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# der: regla de la cadena paso a paso
ax = axes[1]
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("Regla de la Cadena (Chain Rule)", fontsize=11)

steps = [
    (1.0, "x", CYAN),
    (3.0, "z=Wx+b", BLUE),
    (5.5, "a=σ(z)", PURPLE),
    (8.0, "L=loss(a,y)", RED),
]
for xp, lbl, col in steps:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (xp - 0.8, 2.3),
            1.6,
            1.1,
            boxstyle="round,pad=0.1",
            facecolor=col,
            alpha=0.85,
            zorder=3,
        )
    )
    ax.text(
        xp,
        2.85,
        lbl,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )

# flechas forward
for i in range(len(steps) - 1):
    x0 = steps[i][0] + 0.8
    x1 = steps[i + 1][0] - 0.8
    ax.annotate(
        "",
        xy=(x1, 2.9),
        xytext=(x0, 2.9),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=GREEN),
    )

# flechas backward
back_labels = ["∂L/∂a", "∂L/∂z = ∂L/∂a · σ'(z)", "∂L/∂W = ∂L/∂z · x"]
for i, lbl in enumerate(back_labels):
    x0 = steps[i + 1][0] - 0.8
    x1 = steps[i][0] + 0.8
    ax.annotate(
        "",
        xy=(x1, 2.1),
        xytext=(x0, 2.1),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=ORANGE),
    )
    ax.text((x0 + x1) / 2, 1.7, lbl, ha="center", fontsize=7.5, color=ORANGE)

ax.text(5.0, 5.2, "Forward →", ha="center", color=GREEN, fontsize=10, fontweight="bold")
ax.text(
    5.0,
    0.5,
    "← Backward (gradientes)",
    ha="center",
    color=ORANGE,
    fontsize=10,
    fontweight="bold",
)

savefig("04-backprop-gradientes.png")

# ── 05 Funciones de perdida ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG)
fig.suptitle("Funciones de Pérdida Comunes", fontsize=13, fontweight="bold")

# MSE vs MAE vs Huber
ax = axes[0]
ax.set_facecolor(BG)
err = np.linspace(-3, 3, 300)
mse = err**2
mae = np.abs(err)
delta = 1.0
huber = np.where(
    np.abs(err) <= delta, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta)
)
ax.plot(err, mse, label="MSE", color=BLUE, lw=2)
ax.plot(err, mae, label="MAE", color=ORANGE, lw=2)
ax.plot(err, huber, label="Huber", color=GREEN, lw=2, linestyle="--")
ax.set_title("Regresión")
ax.set_xlabel("y - ŷ")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Binary Cross-Entropy
ax = axes[1]
ax.set_facecolor(BG)
p = np.linspace(0.01, 0.99, 300)
bce_pos = -np.log(p)  # y=1
bce_neg = -np.log(1 - p)  # y=0
ax.plot(p, bce_pos, label="BCE (y=1)", color=BLUE, lw=2)
ax.plot(p, bce_neg, label="BCE (y=0)", color=RED, lw=2)
ax.set_title("Clasificación Binaria (BCE)")
ax.set_xlabel("ŷ (prob predicha)")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 5)

# Categorical Cross-Entropy vs clases
ax = axes[2]
ax.set_facecolor(BG)
p_correct = np.linspace(0.01, 0.99, 300)
cce = -np.log(p_correct)
ax.plot(p_correct, cce, color=PURPLE, lw=2.5)
ax.fill_between(p_correct, cce, alpha=0.1, color=PURPLE)
ax.axvline(0.5, color=GRAY, linestyle="--", alpha=0.6)
ax.text(0.52, 3.5, "p=0.5\nL≈0.69", fontsize=9, color=GRAY)
ax.set_title("Categorical Cross-Entropy")
ax.set_xlabel("P(clase correcta)")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3)

savefig("05-funciones-perdida.png")

# ── 06 Curvas de entrenamiento ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG)
fig.suptitle("Diagnóstico por Curvas de Entrenamiento", fontsize=13, fontweight="bold")
epochs = np.arange(1, 51)

scenarios = [
    (
        "Underfitting\n(capacidad baja)",
        0.7 - 0.1 * np.log(epochs) / np.log(50),
        0.75 - 0.09 * np.log(epochs) / np.log(50),
        ORANGE,
        "Ambas pérdidas altas",
    ),
    (
        "Buen ajuste",
        0.7 - 0.4 * np.log(epochs) / np.log(50) + 0.02 * np.random.randn(50),
        0.75 - 0.35 * np.log(epochs) / np.log(50) + 0.025 * np.random.randn(50),
        GREEN,
        "Ambas convergen juntas",
    ),
    (
        "Overfitting",
        0.7 - 0.5 * np.log(epochs) / np.log(50) + 0.01 * np.random.randn(50),
        0.75
        - 0.2 * np.log(epochs) / np.log(50)
        + 0.2 * (epochs / 50) ** 2
        + 0.02 * np.random.randn(50),
        RED,
        "Val sube, train baja",
    ),
]

for ax, (title, tr, va, color, note) in zip(axes, scenarios):
    ax.set_facecolor(BG)
    ax.plot(epochs, np.clip(tr, 0.1, 1), color=BLUE, lw=2, label="Train loss")
    ax.plot(
        epochs, np.clip(va, 0.1, 1), color=color, lw=2, label="Val loss", linestyle="--"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        25,
        ax.get_ylim()[1] * 0.92,
        note,
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=color, alpha=0.8),
    )

savefig("06-curvas-entrenamiento.png")

# ── 07 Decision boundary MLP vs Lineal ──────────────────────────────────────
from sklearn.datasets import make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor=BG)
fig.suptitle("Frontera de Decisión: Lineal vs MLP", fontsize=13, fontweight="bold")

datasets = [
    ("Lunas (no lineal)", *make_moons(n_samples=200, noise=0.2, random_state=42)),
    ("Círculos", *make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)),
]

for row, (name, X, y) in enumerate(datasets):
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for col, (clf_name, clf, color) in enumerate(
        [
            ("Regresión Logística", LogisticRegression(), BLUE),
            (
                "MLP (1 capa oculta)",
                MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42),
                GREEN,
            ),
            (
                "MLP (2 capas ocultas)",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42
                ),
                PURPLE,
            ),
        ]
    ):
        ax = axes[row][col]
        ax.set_facecolor(BG)
        clf.fit(X_s, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(
            X_s[:, 0], X_s[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20, zorder=5
        )
        acc = clf.score(X_s, y)
        ax.set_title(f"{clf_name}\n{name} | Acc={acc:.2f}", fontsize=8)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

savefig("07-decision-boundary.png")

# ── 08 Efecto de profundidad y ancho ─────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle(
    "Profundidad y Ancho de Red: Impacto en Rendimiento", fontsize=13, fontweight="bold"
)

X_c, y_c = make_classification(
    n_samples=800, n_features=10, n_informative=6, n_redundant=2, random_state=42
)
sc2 = StandardScaler()
X_cs = sc2.fit_transform(X_c)

# variando profundidad (capas)
ax = axes[0]
ax.set_facecolor(BG)
depths = [1, 2, 3, 4, 5]
scores_depth = []
for d in depths:
    layers = tuple([32] * d)
    clf = MLPClassifier(hidden_layer_sizes=layers, max_iter=500, random_state=42)
    s = cross_val_score(clf, X_cs, y_c, cv=5, scoring="accuracy")
    scores_depth.append(s.mean())

ax.plot(depths, scores_depth, "o-", color=BLUE, lw=2, markersize=7)
ax.fill_between(
    depths,
    [s - 0.02 for s in scores_depth],
    [s + 0.02 for s in scores_depth],
    alpha=0.15,
    color=BLUE,
)
ax.set_xlabel("Número de capas ocultas (ancho=32)")
ax.set_ylabel("Accuracy (CV)")
ax.set_title("Efecto de la Profundidad")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.7, 1.0)
best_d = depths[np.argmax(scores_depth)]
ax.axvline(
    best_d, color=GREEN, linestyle="--", alpha=0.7, label=f"Mejor: {best_d} capas"
)
ax.legend()

# variando ancho (neuronas)
ax = axes[1]
ax.set_facecolor(BG)
widths = [4, 8, 16, 32, 64, 128, 256]
scores_w = []
for w in widths:
    clf = MLPClassifier(hidden_layer_sizes=(w, w), max_iter=500, random_state=42)
    s = cross_val_score(clf, X_cs, y_c, cv=5, scoring="accuracy")
    scores_w.append(s.mean())

ax.semilogx(widths, scores_w, "s-", color=PURPLE, lw=2, markersize=7)
ax.fill_between(
    widths,
    [s - 0.02 for s in scores_w],
    [s + 0.02 for s in scores_w],
    alpha=0.15,
    color=PURPLE,
)
ax.set_xlabel("Neuronas por capa oculta (log scale)")
ax.set_ylabel("Accuracy (CV)")
ax.set_title("Efecto del Ancho")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.7, 1.0)
best_w = widths[np.argmax(scores_w)]
ax.axvline(
    best_w, color=GREEN, linestyle="--", alpha=0.7, label=f"Mejor: {best_w} neur."
)
ax.legend()

savefig("08-profundidad-ancho.png")

# ── 09 Dashboard resumen ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
fig.suptitle(
    "Dashboard: Redes Neuronales — Conceptos Clave", fontsize=14, fontweight="bold"
)
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# A) Activaciones ReLU vs Sigmoid
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor(BG)
x_ = np.linspace(-4, 4, 200)
ax.plot(x_, 1 / (1 + np.exp(-x_)), color=BLUE, lw=2, label="Sigmoid")
ax.plot(x_, np.maximum(0, x_), color=GREEN, lw=2, label="ReLU")
ax.plot(x_, np.tanh(x_), color=ORANGE, lw=2, label="Tanh")
ax.set_title("Activaciones")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# B) Curva de loss (buen entrenamiento)
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(BG)
ep = np.arange(1, 61)
tr_loss = 1.2 * np.exp(-0.05 * ep) + 0.15 + 0.01 * np.random.randn(60)
va_loss = 1.3 * np.exp(-0.045 * ep) + 0.2 + 0.015 * np.random.randn(60)
ax.plot(ep, tr_loss, color=BLUE, lw=2, label="Train")
ax.plot(ep, va_loss, color=ORANGE, lw=2, label="Val", linestyle="--")
ax.set_title("Curvas de Loss")
ax.set_xlabel("Epoch")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# C) Decision boundary (lunas, MLP)
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor(BG)
X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=7)
sc_ = StandardScaler()
X_ms = sc_.fit_transform(X_m)
mlp_ = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42).fit(
    X_ms, y_m
)
xx_, yy_ = np.meshgrid(np.linspace(-2.5, 2.5, 150), np.linspace(-2, 2, 150))
Z_ = mlp_.predict(np.c_[xx_.ravel(), yy_.ravel()]).reshape(xx_.shape)
ax.contourf(xx_, yy_, Z_, alpha=0.3, cmap="coolwarm")
ax.scatter(
    X_ms[:, 0], X_ms[:, 1], c=y_m, cmap="coolwarm", edgecolor="k", s=15, zorder=5
)
ax.set_title(f"MLP — Lunas Acc={mlp_.score(X_ms,y_m):.2f}")

# D) BCE loss
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor(BG)
p_ = np.linspace(0.01, 0.99, 200)
ax.plot(p_, -np.log(p_), color=BLUE, lw=2, label="BCE (y=1)")
ax.plot(p_, -np.log(1 - p_), color=RED, lw=2, label="BCE (y=0)")
ax.set_title("Binary Cross-Entropy")
ax.set_xlabel("ŷ")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 5)

# E) GD superficie
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor(BG)
w1_ = np.linspace(-3, 3, 150)
w2_ = np.linspace(-3, 3, 150)
W1_, W2_ = np.meshgrid(w1_, w2_)
Z_s = W1_**2 + 3 * W2_**2
c_ = ax.contourf(W1_, W2_, Z_s, levels=15, cmap="Blues", alpha=0.7)
ax.plot(wpath[:, 0], wpath[:, 1], "o-", color=RED, markersize=3, lw=1.5)
ax.set_title("GD en Loss Surface")
ax.set_xlabel("w₁")
ax.set_ylabel("w₂")

# F) Profundidad vs accuracy
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor(BG)
ax.bar(
    depths,
    scores_depth,
    color=[GREEN if d == best_d else BLUE for d in depths],
    alpha=0.8,
)
ax.set_xlabel("Capas ocultas")
ax.set_ylabel("Accuracy")
ax.set_title("Profundidad vs Acc")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0.7, 1.0)

savefig("09-dashboard.png")

print("✓ Todos los graficos de tema-10 generados.")
