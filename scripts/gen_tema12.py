"""Genera 9 graficos para tema-12: Tecnicas de Entrenamiento en Deep Learning."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings("ignore")

import os

OUT = "public/ruta-aprendizaje-graficos/tema-12"
os.makedirs(OUT, exist_ok=True)
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


# ── 01 Comparativa de optimizadores en cuenco 2D ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig.suptitle(
    "Optimizadores: Trayectorias en Funcion de Perdida 2D",
    fontsize=13,
    fontweight="bold",
)


# Funcion bowl elongada (simula mal condicionamiento)
def loss_fn(w1, w2):
    return 0.5 * w1**2 + 5 * w2**2


# Simular trayectorias simplificadas
def sgd_path(start, lr, steps=40):
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        g = np.array([w[0], 10 * w[1]])
        g += np.random.randn(2) * 0.05
        w -= lr * g
        path.append(w.copy())
    return np.array(path)


def adam_path(start, lr, steps=40, b1=0.9, b2=0.999, eps=1e-8):
    w = np.array(start, dtype=float)
    m, v = np.zeros(2), np.zeros(2)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = np.array([w[0], 10 * w[1]])
        g += np.random.randn(2) * 0.05
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return np.array(path)


def momentum_path(start, lr, steps=40, mu=0.9):
    w = np.array(start, dtype=float)
    vel = np.zeros(2)
    path = [w.copy()]
    for _ in range(steps):
        g = np.array([w[0], 10 * w[1]])
        g += np.random.randn(2) * 0.05
        vel = mu * vel - lr * g
        w += vel
        path.append(w.copy())
    return np.array(path)


start = [2.5, 0.8]
configs = [
    ("SGD (lr=0.1)", sgd_path(start, 0.1), ORANGE),
    ("SGD + Momentum", momentum_path(start, 0.05), BLUE),
    ("Adam (lr=0.3)", adam_path(start, 0.3), GREEN),
]

W1 = np.linspace(-3, 3, 200)
W2 = np.linspace(-1, 1, 200)
W1g, W2g = np.meshgrid(W1, W2)
Z = loss_fn(W1g, W2g)

for ax, (title, path, color) in zip(axes, configs):
    ax.set_facecolor(BG)
    ax.contourf(W1g, W2g, Z, levels=20, cmap="Blues", alpha=0.6)
    ax.contour(W1g, W2g, Z, levels=10, colors="white", linewidths=0.5, alpha=0.4)
    ax.plot(path[:, 0], path[:, 1], "o-", color=color, ms=4, lw=1.8, label=title)
    ax.plot(*start, "s", color=RED, ms=9, zorder=5, label="inicio")
    ax.plot(0, 0, "*", color="yellow", ms=14, zorder=5, label="optimo")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.legend(fontsize=7, loc="upper right")

savefig("01-optimizadores-trayectorias.png")


# ── 02 Learning Rate Schedules ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=BG)
fig.suptitle("Learning Rate Schedules", fontsize=13, fontweight="bold")

epochs = np.arange(0, 100)
lr0 = 0.1

# Step decay
step_lr = lr0 * (0.5 ** (epochs // 20))

# Exponential decay
exp_lr = lr0 * np.exp(-0.03 * epochs)

# Cosine annealing
cos_lr = lr0 / 2 * (1 + np.cos(np.pi * epochs / 100))

# Warmup + cosine
warmup_ep = 10
cos_part = lr0 / 2 * (1 + np.cos(np.pi * (epochs - warmup_ep) / (100 - warmup_ep)))
warmup_part = lr0 * epochs / warmup_ep
warmup_cos = np.where(epochs < warmup_ep, warmup_part, cos_part)

schedules = [
    (axes[0, 0], step_lr, BLUE, "Step Decay (drop×0.5 cada 20 ep)"),
    (axes[0, 1], exp_lr, ORANGE, "Exponential Decay (γ=0.97)"),
    (axes[1, 0], cos_lr, GREEN, "Cosine Annealing"),
    (axes[1, 1], warmup_cos, PURPLE, "Linear Warmup + Cosine"),
]

for ax, lr, c, title in schedules:
    ax.set_facecolor(BG)
    ax.plot(epochs, lr, color=c, lw=2.5)
    ax.fill_between(epochs, lr, alpha=0.15, color=c)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Learning Rate")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

savefig("02-lr-schedules.png")


# ── 03 Dropout: efecto en overfitting ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Efecto de Dropout en Overfitting", fontsize=13, fontweight="bold")

epochs_d = np.arange(1, 81)

# Sin dropout: gap enorme
train_no = 0.9 - 0.85 * np.exp(-0.08 * epochs_d) + np.random.randn(80) * 0.005
val_no = (
    0.85
    - 0.60 * np.exp(-0.05 * epochs_d)
    - 0.0003 * epochs_d
    + np.random.randn(80) * 0.008
)
val_no = np.clip(val_no, 0.4, 1)

# Con dropout: curvas mas juntas
train_dp = 0.88 - 0.78 * np.exp(-0.07 * epochs_d) + np.random.randn(80) * 0.006
val_dp = 0.86 - 0.74 * np.exp(-0.065 * epochs_d) + np.random.randn(80) * 0.007

for ax, (tr, vl, title, dp) in zip(
    axes,
    [
        (train_no, val_no, "Sin Dropout", False),
        (train_dp, val_dp, "Con Dropout (p=0.3)", True),
    ],
):
    ax.set_facecolor(BG)
    ax.plot(epochs_d, tr, color=BLUE, lw=2, label="Train")
    ax.plot(epochs_d, vl, color=ORANGE, lw=2, label="Validacion")
    if not dp:
        ax.fill_between(
            epochs_d, tr, vl, alpha=0.15, color=RED, label="Gap (overfitting)"
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

savefig("03-dropout-overfitting.png")


# ── 04 Batch Normalization: distribucion de activaciones ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor=BG)
fig.suptitle(
    "Batch Normalization: Distribucion de Activaciones por Capa",
    fontsize=13,
    fontweight="bold",
)

layers = ["Capa 1", "Capa 2", "Capa 3"]

# Sin BN: la distribución se desplaza cada capa (internal covariate shift)
for i, (ax, layer) in enumerate(zip(axes[0], layers)):
    ax.set_facecolor(BG)
    shift = i * 1.5
    scale = 1 + i * 0.8
    x = np.linspace(-5, 12, 300)
    y = np.exp(-0.5 * ((x - shift) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
    ax.plot(x, y, color=ORANGE, lw=2)
    ax.fill_between(x, y, alpha=0.3, color=ORANGE)
    ax.set_title(f"Sin BN — {layer}\nμ={shift:.1f}, σ={scale:.1f}", fontsize=9)
    ax.set_xlabel("Activacion")
    ax.set_ylabel("Densidad")
    ax.grid(True, alpha=0.3)

# Con BN: distribución centrada y estable en cada capa
for i, (ax, layer) in enumerate(zip(axes[1], layers)):
    ax.set_facecolor(BG)
    x = np.linspace(-4, 4, 300)
    y = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    ax.plot(x, y, color=GREEN, lw=2)
    ax.fill_between(x, y, alpha=0.3, color=GREEN)
    ax.set_title(f"Con BN — {layer}\nμ≈0, σ≈1", fontsize=9)
    ax.set_xlabel("Activacion")
    ax.set_ylabel("Densidad")
    ax.grid(True, alpha=0.3)

axes[0, 0].text(
    -0.15,
    0.5,
    "Sin\nBatchNorm",
    transform=axes[0, 0].transAxes,
    fontsize=10,
    va="center",
    rotation=90,
    fontweight="bold",
    color=ORANGE,
)
axes[1, 0].text(
    -0.15,
    0.5,
    "Con\nBatchNorm",
    transform=axes[1, 0].transAxes,
    fontsize=10,
    va="center",
    rotation=90,
    fontweight="bold",
    color=GREEN,
)

savefig("04-batch-normalization.png")


# ── 05 Early Stopping ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(BG)
fig.suptitle(
    "Early Stopping: Detectar el Punto Optimo de Generalizacion",
    fontsize=13,
    fontweight="bold",
)

ep = np.arange(1, 121)
train_loss = 1.8 * np.exp(-0.04 * ep) + 0.05 + np.random.randn(120) * 0.008
# val mejora y luego empeora
val_loss = (
    1.8 * np.exp(-0.025 * ep)
    + 0.15
    + 0.0018 * (ep - 60) ** 2 / 60 * (ep > 60)
    + np.random.randn(120) * 0.015
)
val_loss = np.clip(val_loss, 0.1, 2)
best_ep = np.argmin(val_loss) + 1

ax.plot(ep, train_loss, color=BLUE, lw=2, label="Loss de Entrenamiento")
ax.plot(ep, val_loss, color=ORANGE, lw=2, label="Loss de Validacion")
ax.axvline(best_ep, color=GREEN, lw=2.5, ls="--", label=f"Mejor epoca ({best_ep})")
ax.axvspan(best_ep, 120, alpha=0.07, color=RED, label="Zona de overfitting")
ax.annotate(
    f"Guardar\ncheckpoint\naqu\u00ed",
    xy=(best_ep, val_loss[best_ep - 1]),
    xytext=(best_ep + 10, val_loss[best_ep - 1] + 0.15),
    arrowprops=dict(arrowstyle="->", color=GREEN),
    fontsize=9,
    color=GREEN,
)
ax.set_xlabel("Epoca", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

savefig("05-early-stopping.png")


# ── 06 Inicializacion de pesos ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 5), facecolor=BG)
fig.suptitle(
    "Inicializacion de Pesos: Distribucion Inicial", fontsize=13, fontweight="bold"
)

n_in, n_out = 512, 256

uniform = np.random.uniform(-0.1, 0.1, (n_in, n_out)).flatten()
normal = np.random.randn(n_in * n_out) * 0.01
xavier = np.random.randn(n_in * n_out) * np.sqrt(2 / (n_in + n_out))  # Glorot
he = np.random.randn(n_in * n_out) * np.sqrt(2 / n_in)  # He/Kaiming

inits = [
    (axes[0], uniform, ORANGE, "Uniforme [-0.1, 0.1]", "Puede saturar\nsigmoid/tanh"),
    (axes[1], normal, BLUE, "Normal (μ=0, σ=0.01)", "Muy pequena:\ngradientes debiles"),
    (axes[2], xavier, GREEN, "Xavier / Glorot", "Para sigmoid/tanh\n+ redes moderadas"),
    (axes[3], he, PURPLE, "He / Kaiming", "Para ReLU\n(recomendada)"),
]

for ax, data, c, title, note in inits:
    ax.set_facecolor(BG)
    ax.hist(data, bins=60, color=c, alpha=0.75, edgecolor="white", lw=0.3)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Valor del peso")
    ax.set_ylabel("Frecuencia")
    ax.text(
        0.5,
        0.92,
        note,
        transform=ax.transAxes,
        fontsize=7.5,
        ha="center",
        va="top",
        color=GRAY,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ax.grid(True, alpha=0.2)

savefig("06-inicializacion-pesos.png")


# ── 07 Gradient Clipping ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle(
    "Gradient Clipping: Control de Gradientes Explosivos",
    fontsize=13,
    fontweight="bold",
)

# Simular normas de gradientes durante entrenamiento
ep7 = np.arange(1, 101)
grad_norms = 0.5 + 0.3 * np.sin(0.3 * ep7) + np.abs(np.random.randn(100) * 0.4)
# Agregar explosiones
grad_norms[15] = 8.5
grad_norms[38] = 11.2
grad_norms[62] = 7.8
grad_norms[81] = 9.4

clip_val = 1.0
clipped = np.clip(grad_norms, 0, clip_val)

# Izquierda: sin clipping
ax = axes[0]
ax.set_facecolor(BG)
ax.bar(
    ep7,
    grad_norms,
    color=np.where(grad_norms > clip_val, RED, BLUE),
    width=1,
    alpha=0.8,
)
ax.axhline(clip_val, color=ORANGE, lw=2, ls="--", label=f"Clip threshold = {clip_val}")
ax.set_title("Sin Gradient Clipping", fontsize=11, fontweight="bold")
ax.set_xlabel("Iteracion")
ax.set_ylabel("Norma del gradiente ||g||")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.text(
    0.5,
    0.85,
    "Picos = entrenamiento\ninestable",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    color=RED,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor=RED, lw=1),
)

# Derecha: con clipping
ax = axes[1]
ax.set_facecolor(BG)
ax.bar(ep7, clipped, color=GREEN, width=1, alpha=0.8)
ax.axhline(clip_val, color=ORANGE, lw=2, ls="--", label=f"Clip threshold = {clip_val}")
ax.set_title("Con Gradient Clipping (max_norm=1.0)", fontsize=11, fontweight="bold")
ax.set_xlabel("Iteracion")
ax.set_ylabel("Norma del gradiente ||g||")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

savefig("07-gradient-clipping.png")


# ── 08 Comparativa completa: train/val con distintas tecnicas ────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=BG)
fig.suptitle(
    "Impacto Combinado de Tecnicas de Entrenamiento", fontsize=13, fontweight="bold"
)

ep8 = np.arange(1, 101)

# Baseline sin nada
tr_base = 0.9 * (1 - np.exp(-0.05 * ep8)) + np.random.randn(100) * 0.008
vl_base = (
    0.65 * (1 - np.exp(-0.04 * ep8))
    - 0.001 * ep8 * (ep8 > 40)
    + np.random.randn(100) * 0.012
)
vl_base = np.clip(vl_base, 0.1, 0.9)

# Con BN + He init
tr_bn = 0.94 * (1 - np.exp(-0.07 * ep8)) + np.random.randn(100) * 0.007
vl_bn = 0.82 * (1 - np.exp(-0.065 * ep8)) + np.random.randn(100) * 0.009

# Con Dropout + weight decay
tr_dp = 0.91 * (1 - np.exp(-0.06 * ep8)) + np.random.randn(100) * 0.007
vl_dp = 0.87 * (1 - np.exp(-0.058 * ep8)) + np.random.randn(100) * 0.008

# Con todo + cosine LR
tr_all = 0.95 * (1 - np.exp(-0.075 * ep8)) + np.random.randn(100) * 0.006
vl_all = 0.93 * (1 - np.exp(-0.072 * ep8)) + np.random.randn(100) * 0.007

scenarios = [
    (axes[0, 0], tr_base, vl_base, "Baseline (sin tecnicas)", ORANGE),
    (axes[0, 1], tr_bn, vl_bn, "BN + He Initialization", BLUE),
    (axes[1, 0], tr_dp, vl_dp, "Dropout + Weight Decay", GREEN),
    (axes[1, 1], tr_all, vl_all, "Todo combinado + Cosine LR", PURPLE),
]

for ax, tr, vl, title, c in scenarios:
    ax.set_facecolor(BG)
    ax.plot(ep8, tr, color=c, lw=2, label="Train Acc")
    ax.plot(ep8, vl, color=c, lw=2, ls="--", alpha=0.7, label="Val Acc")
    ax.fill_between(ep8, tr, vl, alpha=0.08, color=c)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    final_gap = abs(tr[-1] - vl[-1])
    ax.text(
        0.98,
        0.05,
        f"Gap final: {final_gap:.3f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=c,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor=c, lw=1),
    )

savefig("08-tecnicas-combinadas.png")


# ── 09 Dashboard resumen ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
fig.suptitle(
    "Dashboard: Tecnicas de Entrenamiento en Deep Learning",
    fontsize=14,
    fontweight="bold",
)
gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

ax_opt = fig.add_subplot(gs[0, :2])
ax_lr = fig.add_subplot(gs[0, 2:])
ax_reg = fig.add_subplot(gs[1, :2])
ax_diag = fig.add_subplot(gs[1, 2:])
ax_tbl = fig.add_subplot(gs[2, :])

for ax in [ax_opt, ax_lr, ax_reg, ax_diag, ax_tbl]:
    ax.set_facecolor(BG)

# Mini: optimizadores
ep_d = np.arange(1, 51)
sgd_conv = 1.5 * np.exp(-0.03 * ep_d) + 0.3 + np.random.randn(50) * 0.02
adam_conv = 1.5 * np.exp(-0.08 * ep_d) + 0.05 + np.random.randn(50) * 0.015
adamw_conv = 1.5 * np.exp(-0.085 * ep_d) + 0.04 + np.random.randn(50) * 0.012
ax_opt.plot(ep_d, sgd_conv, color=ORANGE, lw=2, label="SGD")
ax_opt.plot(ep_d, adam_conv, color=BLUE, lw=2, label="Adam")
ax_opt.plot(ep_d, adamw_conv, color=GREEN, lw=2, label="AdamW")
ax_opt.set_title("Convergencia: SGD vs Adam vs AdamW", fontsize=9, fontweight="bold")
ax_opt.set_xlabel("Epoca")
ax_opt.set_ylabel("Loss")
ax_opt.legend(fontsize=8)
ax_opt.grid(True, alpha=0.3)

# Mini: schedules
ax_lr.plot(ep_d, 0.1 * np.exp(-0.04 * ep_d), color=ORANGE, lw=2, label="Exponential")
ax_lr.plot(
    ep_d, 0.05 * (1 + np.cos(np.pi * ep_d / 50)), color=BLUE, lw=2, label="Cosine"
)
ax_lr.plot(ep_d, 0.1 * (0.5 ** (ep_d // 10)), color=GREEN, lw=2, label="Step")
ax_lr.set_title("Learning Rate Schedules", fontsize=9, fontweight="bold")
ax_lr.set_xlabel("Epoca")
ax_lr.set_ylabel("LR")
ax_lr.legend(fontsize=8)
ax_lr.grid(True, alpha=0.3)

# Mini: early stopping
ep_es = np.arange(1, 61)
vl_es = (
    0.8 * np.exp(-0.06 * ep_es)
    + 0.12
    + 0.002 * (ep_es - 35) ** 2 / 35 * (ep_es > 35)
    + np.random.randn(60) * 0.01
)
best = np.argmin(vl_es) + 1
ax_reg.plot(ep_es, vl_es, color=PURPLE, lw=2)
ax_reg.axvline(best, color=GREEN, ls="--", lw=2, label=f"Early stop (ep {best})")
ax_reg.set_title("Early Stopping", fontsize=9, fontweight="bold")
ax_reg.set_xlabel("Epoca")
ax_reg.set_ylabel("Val Loss")
ax_reg.legend(fontsize=8)
ax_reg.grid(True, alpha=0.3)

# Mini: checklist diagnostico
ax_diag.axis("off")
ax_diag.set_title("Checklist de Diagnostico", fontsize=9, fontweight="bold")
checks = [
    ("✓  train↓ val↓", GREEN, "Correcto"),
    ("✗  train↓ val↑", RED, "Overfitting → dropout/L2"),
    ("✗  train↑ val↑", ORANGE, "Underfitting → red mas grande"),
    ("✗  Loss oscila", PURPLE, "LR alto → reducir LR"),
    ("✗  Loss no baja", CYAN, "LR bajo / mala init"),
]
for i, (text, c, label) in enumerate(checks):
    y = 0.85 - i * 0.18
    ax_diag.add_patch(
        FancyBboxPatch(
            (0.02, y - 0.07),
            0.96,
            0.14,
            boxstyle="round,pad=0.01",
            facecolor=c,
            alpha=0.2,
            edgecolor=c,
            lw=1.5,
            transform=ax_diag.transAxes,
        )
    )
    ax_diag.text(
        0.05,
        y,
        text,
        transform=ax_diag.transAxes,
        fontsize=8.5,
        va="center",
        fontweight="bold",
        color=c,
    )
    ax_diag.text(
        0.95,
        y,
        label,
        transform=ax_diag.transAxes,
        fontsize=7.5,
        va="center",
        ha="right",
        color=GRAY,
    )

# Tabla resumen
ax_tbl.axis("off")
headers = ["Tecnica", "Cuando usar", "Efecto principal", "Libreria"]
rows = [
    [
        "SGD + Momentum",
        "Siempre como baseline",
        "Estable, generaliza bien",
        "torch.optim.SGD(momentum=0.9)",
    ],
    [
        "Adam / AdamW",
        "Default en DL moderno",
        "Converge rapido, lr adaptativo",
        "torch.optim.AdamW(weight_decay=1e-2)",
    ],
    [
        "Cosine LR",
        "Entrenamientos largos",
        "Suaviza convergencia final",
        "CosineAnnealingLR",
    ],
    [
        "Dropout (p=0.3-0.5)",
        "Capas densas, overfit alto",
        "Regulariza, reduce coadaptacion",
        "nn.Dropout(p=0.3)",
    ],
    [
        "Batch Norm",
        "CNNs y MLPs profundos",
        "Estabiliza, acelera training",
        "nn.BatchNorm1d/2d",
    ],
    [
        "Early Stopping",
        "Siempre",
        "Evita overfitting por exceso ep",
        "Guarda mejor val checkpoint",
    ],
    [
        "He Initialization",
        "Redes con ReLU",
        "Gradientes bien escalados",
        "nn.init.kaiming_normal_",
    ],
    [
        "Grad Clipping",
        "RNNs, transformers",
        "Evita explosión de gradientes",
        "clip_grad_norm_(params, 1.0)",
    ],
]
table = ax_tbl.table(
    cellText=rows,
    colLabels=headers,
    cellLoc="left",
    loc="center",
    colWidths=[0.18, 0.22, 0.28, 0.32],
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.4)
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor("#e8eaf6" if r == 0 else (BG if r % 2 == 0 else "#f3f4f6"))
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_text_props(fontweight="bold", color="#1a237e")

savefig("09-dashboard.png")

print("Todos los graficos generados en", OUT)
