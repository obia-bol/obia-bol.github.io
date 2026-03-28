"""Genera 9 graficos para tema-11: Fundamentos de PyTorch."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings("ignore")

OUT = "public/ruta-aprendizaje-graficos/tema-11"
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


# ── 01 Tensores: shapes, operaciones y broadcasting ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=BG)
fig.suptitle("Tensores PyTorch: Shapes y Operaciones", fontsize=14, fontweight="bold")

# (0,0) Escalar, vector, matriz, tensor 3D — visualizacion de shapes
ax = axes[0, 0]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Dimensiones de un Tensor", fontsize=10)
shapes = [
    (0.15, 0.70, "Escalar\nshape: []", BLUE, 0.08, 0.08),
    (0.15, 0.38, "Vector\nshape: [4]", ORANGE, 0.32, 0.08),
    (0.55, 0.65, "Matriz\nshape: [3,4]", GREEN, 0.32, 0.22),
    (0.55, 0.18, "Tensor 3D\nshape: [2,3,4]", PURPLE, 0.32, 0.22),
]
for x, y, label, c, w, h in shapes:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01",
            facecolor=c,
            alpha=0.7,
            edgecolor="white",
            lw=2,
            transform=ax.transAxes,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
        transform=ax.transAxes,
    )
# profundidad del tensor 3D
ax.add_patch(
    FancyBboxPatch(
        (0.60, 0.22),
        0.32,
        0.22,
        boxstyle="round,pad=0.01",
        facecolor=PURPLE,
        alpha=0.4,
        edgecolor="white",
        lw=1,
        transform=ax.transAxes,
    )
)
ax.text(
    0.76,
    0.12,
    "(2 capas)",
    ha="center",
    fontsize=8,
    color=PURPLE,
    transform=ax.transAxes,
)

# (0,1) Broadcasting: (3,1) + (1,4) → (3,4)
ax = axes[0, 1]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Broadcasting (3,1) + (1,4) → (3,4)", fontsize=10)
A = np.array([[1], [2], [3]])
B = np.array([[10, 20, 30, 40]])
C = A + B


# dibujar matrices como tablas
def draw_matrix(ax, M, x0, y0, cell=0.18, color=BLUE, label=""):
    r, c_ = M.shape
    for i in range(r):
        for j in range(c_):
            ax.add_patch(
                plt.Rectangle(
                    (x0 + j * cell, y0 - i * cell),
                    cell * 0.92,
                    cell * 0.88,
                    facecolor=color,
                    alpha=0.6 + 0.1 * ((i + j) % 2 == 0),
                    edgecolor="white",
                    lw=1.5,
                )
            )
            ax.text(
                x0 + j * cell + cell * 0.46,
                y0 - i * cell + cell * 0.44,
                str(M[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
    if label:
        ax.text(
            x0 + c_ * cell / 2,
            y0 + cell * 1.1,
            label,
            ha="center",
            fontsize=9,
            color=GRAY,
        )


draw_matrix(ax, A, 0.05, 0.7, cell=0.17, color=ORANGE, label="A (3,1)")
ax.text(0.30, 0.54, "+", fontsize=20, ha="center", va="center", color=GRAY)
draw_matrix(ax, B, 0.38, 0.7, cell=0.13, color=CYAN, label="B (1,4)")
ax.text(
    0.05 + 4 * 0.13 / 2 + 0.38,
    0.40,
    "=",
    fontsize=20,
    ha="center",
    va="center",
    color=GRAY,
)
draw_matrix(ax, C, 0.05, 0.28, cell=0.13, color=BLUE, label="C (3,4)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# (0,2) Operaciones comunes: suma, producto, reshape
ax = axes[0, 2]
ax.set_facecolor(BG)
ax.set_title("Operaciones Elementales", fontsize=10)
ops = [
    "torch.add(a,b)\na + b",
    "torch.mul(a,b)\na * b",
    "torch.matmul(A,B)\nA @ B",
    "a.reshape(r,c)",
    "a.permute(2,0,1)",
    "a.squeeze()\na.unsqueeze(0)",
    "torch.cat([a,b],dim=0)",
    "torch.stack([a,b])",
    "a.float()\na.to(device)",
]
colors_ops = [BLUE, BLUE, ORANGE, GREEN, GREEN, CYAN, RED, RED, PURPLE]
for i, (op, c) in enumerate(zip(ops, colors_ops)):
    row, col_ = divmod(i, 3)
    ax.add_patch(
        FancyBboxPatch(
            (col_ * 0.33 + 0.01, 0.88 - row * 0.3),
            0.31,
            0.24,
            boxstyle="round,pad=0.02",
            facecolor=c,
            alpha=0.75,
            edgecolor="white",
            lw=1.5,
        )
    )
    ax.text(
        col_ * 0.33 + 0.165,
        0.88 - row * 0.3 + 0.12,
        op,
        ha="center",
        va="center",
        fontsize=7.5,
        color="white",
        fontweight="bold",
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# (1,0) Memoria: contiguous, view vs reshape
ax = axes[1, 0]
ax.set_facecolor(BG)
ax.set_title("view() vs reshape() vs contiguous()", fontsize=10)
ax.axis("off")
steps = [
    (0.1, 0.75, "t = torch.randn(3,4)\n# shape (3,4), contiguo"),
    (0.1, 0.52, "v = t.view(4,3)\n# OK: mismo bloque de memoria"),
    (0.1, 0.30, "t2 = t.transpose(0,1)\n# shape (4,3), NO contiguo"),
    (0.1, 0.08, "r = t2.contiguous().view(12)\n# hacer contiguo primero"),
]
colors_s = [BLUE, GREEN, ORANGE, PURPLE]
for x, y, txt, c in zip(*zip(*[(s[0], s[1], s[2]) for s in steps]), colors_s):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.82,
            0.18,
            boxstyle="round,pad=0.02",
            facecolor=c,
            alpha=0.25,
            edgecolor=c,
            lw=1.5,
        )
    )
    ax.text(
        x + 0.41,
        y + 0.09,
        txt,
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
        family="monospace",
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# (1,1) dtype y device
ax = axes[1, 1]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("dtype y device", fontsize=10)
dtypes = [
    ("torch.float32", "Default DL", GREEN),
    ("torch.float16", "AMP / GPU", ORANGE),
    ("torch.int64", "Indices/labels", BLUE),
    ("torch.bool", "Mascaras", PURPLE),
]
devices = [
    ("cpu", "Siempre disponible", BLUE),
    ("cuda:0", "GPU Nvidia (CUDA)", GREEN),
    ("mps", "GPU Apple Silicon", ORANGE),
]
for i, (name, desc, c) in enumerate(dtypes):
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.73 - i * 0.17),
            0.44,
            0.14,
            boxstyle="round,pad=0.02",
            facecolor=c,
            alpha=0.6,
            edgecolor="white",
            lw=1,
        )
    )
    ax.text(
        0.25,
        0.73 - i * 0.17 + 0.07,
        f"{name}\n{desc}",
        ha="center",
        va="center",
        fontsize=7.5,
        color="white",
        fontweight="bold",
    )
for i, (name, desc, c) in enumerate(devices):
    ax.add_patch(
        FancyBboxPatch(
            (0.53, 0.73 - i * 0.22),
            0.44,
            0.18,
            boxstyle="round,pad=0.02",
            facecolor=c,
            alpha=0.6,
            edgecolor="white",
            lw=1,
        )
    )
    ax.text(
        0.75,
        0.73 - i * 0.22 + 0.09,
        f"{name}\n{desc}",
        ha="center",
        va="center",
        fontsize=7.5,
        color="white",
        fontweight="bold",
    )
ax.text(0.25, 0.96, "dtypes", ha="center", fontsize=9, fontweight="bold", color=GRAY)
ax.text(0.75, 0.96, "devices", ha="center", fontsize=9, fontweight="bold", color=GRAY)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# (1,2) Gradient computation graph
ax = axes[1, 2]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Grafo Computacional (autograd)", fontsize=10)
nodes = {
    "x\n(leaf)": (0.15, 0.75, CYAN),
    "W\n(leaf)": (0.55, 0.75, CYAN),
    "b\n(leaf)": (0.85, 0.55, CYAN),
    "z=Wx+b": (0.45, 0.52, BLUE),
    "a=relu(z)": (0.45, 0.30, PURPLE),
    "L=MSE(a,y)": (0.45, 0.08, RED),
}
pos = {}
for name, (x, y, c) in nodes.items():
    pos[name] = (x, y)
    ax.add_patch(plt.Circle((x, y), 0.085, color=c, zorder=5, alpha=0.85))
    ax.text(
        x,
        y,
        name,
        ha="center",
        va="center",
        fontsize=7,
        color="white",
        fontweight="bold",
        zorder=6,
    )

edges_fwd = [
    ("x\n(leaf)", "z=Wx+b"),
    ("W\n(leaf)", "z=Wx+b"),
    ("b\n(leaf)", "z=Wx+b"),
    ("z=Wx+b", "a=relu(z)"),
    ("a=relu(z)", "L=MSE(a,y)"),
]
for src, dst in edges_fwd:
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    ax.annotate(
        "",
        xy=(x1, y1 + 0.085),
        xytext=(x0, y0 - 0.085),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=GREEN),
    )

# backward arrows
edges_bwd = [
    ("L=MSE(a,y)", "a=relu(z)"),
    ("a=relu(z)", "z=Wx+b"),
    ("z=Wx+b", "W\n(leaf)"),
]
for src, dst in edges_bwd:
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    ax.annotate(
        "",
        xy=(x1 + 0.10, y1),
        xytext=(x0 + 0.10, y0),
        arrowprops=dict(
            arrowstyle="->", lw=1.2, color=ORANGE, connectionstyle="arc3,rad=0.3"
        ),
    )

ax.text(0.78, 0.55, "Forward →\n(verde)", fontsize=8, color=GREEN, ha="center")
ax.text(0.82, 0.32, "← Backward\n(naranja)", fontsize=8, color=ORANGE, ha="center")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

savefig("01-tensores-shapes.png")

# ── 02 Autograd: requires_grad y .backward() ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Autograd: Gradientes Automáticos", fontsize=13, fontweight="bold")

# izq: calculo numerico vs autograd
ax = axes[0]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Diferenciacion automatica vs numerica", fontsize=10)
rows = [
    (
        "Diferenciacion\nnumerica",
        "(f(x+ε)-f(x))/ε",
        "Solo puntual\nO(n) forward passes",
        ORANGE,
    ),
    (
        "Autograd\n(modo reverso)",
        ".backward()\nuna vez",
        "Gradiente completo\n1 forward + 1 backward",
        GREEN,
    ),
    (
        "Diferenciacion\nsiimbolica",
        "reglas\nalgebraicas",
        "Exacta pero\nexponencialmente grande",
        BLUE,
    ),
]
for i, (name, formula, desc, c) in enumerate(rows):
    y_pos = 0.72 - i * 0.28
    ax.add_patch(
        FancyBboxPatch(
            (0.02, y_pos),
            0.27,
            0.22,
            boxstyle="round,pad=0.03",
            facecolor=c,
            alpha=0.75,
            edgecolor="white",
            lw=2,
        )
    )
    ax.text(
        0.155,
        y_pos + 0.11,
        name,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.31, y_pos),
            0.27,
            0.22,
            boxstyle="round,pad=0.03",
            facecolor=c,
            alpha=0.4,
            edgecolor=c,
            lw=1.5,
        )
    )
    ax.text(
        0.445,
        y_pos + 0.11,
        formula,
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
        family="monospace",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.60, y_pos),
            0.38,
            0.22,
            boxstyle="round,pad=0.03",
            facecolor=c,
            alpha=0.15,
            edgecolor=c,
            lw=1,
        )
    )
    ax.text(
        0.79, y_pos + 0.11, desc, ha="center", va="center", fontsize=8.5, color="black"
    )
for label, x in [("Metodo", 0.155), ("Formula/API", 0.445), ("Observacion", 0.79)]:
    ax.text(x, 0.96, label, ha="center", fontsize=9, fontweight="bold", color=GRAY)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# der: flujo requires_grad
ax = axes[1]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Flujo requires_grad", fontsize=10)
code_lines = [
    ("import torch", GRAY),
    ("", GRAY),
    ("x = torch.tensor([2.0], requires_grad=True)", BLUE),
    ("W = torch.tensor([[3.0]], requires_grad=True)", BLUE),
    ("b = torch.tensor([1.0], requires_grad=True)", BLUE),
    ("", GRAY),
    ("z = W @ x + b   # z = 3*2+1 = 7", GREEN),
    ("L = z.pow(2).mean()  # L = 49", GREEN),
    ("", GRAY),
    ("L.backward()     # calcula dL/dW, dL/db, dL/dx", ORANGE),
    ("", GRAY),
    ("print(W.grad)    # tensor([[28.]])  ← dL/dW = 2*z*x", RED),
    ("print(b.grad)    # tensor([14.])   ← dL/db = 2*z", RED),
    ("print(x.grad)    # tensor([42.])   ← dL/dx = 2*z*W", RED),
    ("", GRAY),
    ("# Limpiar gradientes antes del siguiente paso:", GRAY),
    ("W.grad.zero_()   # in-place", PURPLE),
]
for i, (line, c) in enumerate(code_lines):
    ax.text(
        0.03,
        0.95 - i * 0.056,
        line,
        fontsize=8.2,
        color=c,
        family="monospace",
        va="top",
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

savefig("02-autograd.png")

# ── 03 DataLoader: batching y shuffle ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle(
    "Dataset y DataLoader: Carga Eficiente de Datos", fontsize=13, fontweight="bold"
)

# izq: diagrama Dataset → DataLoader → Model
ax = axes[0]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Pipeline de datos", fontsize=10)
components = [
    (0.5, 0.88, "Datos crudos\n(CSV, imágenes, etc.)", GRAY),
    (0.5, 0.68, "Dataset\n__len__ + __getitem__", CYAN),
    (0.5, 0.48, "DataLoader\nbatch_size, shuffle, num_workers", BLUE),
    (0.25, 0.26, "Train loop\nmodel(batch)", GREEN),
    (0.75, 0.26, "Val loop\ntorch.no_grad()", ORANGE),
    (0.5, 0.06, "Optimizer.step()\nloss.backward()", PURPLE),
]
for x, y, lbl, c in components:
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.22, y - 0.07),
            0.44,
            0.14,
            boxstyle="round,pad=0.02",
            facecolor=c,
            alpha=0.7,
            edgecolor="white",
            lw=2,
        )
    )
    ax.text(
        x,
        y,
        lbl,
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
    )
# flechas
arrows = [
    (0.5, 0.81, 0.5, 0.75),
    (0.5, 0.61, 0.5, 0.55),
    (0.5, 0.41, 0.25, 0.33),
    (0.5, 0.41, 0.75, 0.33),
    (0.25, 0.19, 0.5, 0.13),
    (0.75, 0.19, 0.5, 0.13),
]
for x0, y0, x1, y1 in arrows:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=GRAY),
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# der: efecto del batch size
ax = axes[1]
ax.set_facecolor(BG)
batch_sizes = [8, 16, 32, 64, 128, 256, 512]
n_samples = 50000
# tiempo relativo de actualizacion (proporcional a n_batches * overhead/batch)
# overhead constante por batch + tiempo de computo proporcional a batch_size
n_batches = [n_samples / b for b in batch_sizes]
# noise gradient: inversamente proporcional a sqrt(batch_size)
noise_grad = [1.0 / np.sqrt(b) for b in batch_sizes]
# memoria: proporcional a batch_size
memory = [b / 512 for b in batch_sizes]

ax2 = ax.twinx()
(l1,) = ax.plot(
    range(len(batch_sizes)),
    noise_grad,
    "o-",
    color=RED,
    lw=2.5,
    markersize=8,
    label="Ruido del gradiente (↓ = mejor)",
)
(l2,) = ax.plot(
    range(len(batch_sizes)),
    memory,
    "s--",
    color=BLUE,
    lw=2,
    markersize=7,
    label="Memoria relativa",
)
(l3,) = ax2.plot(
    range(len(batch_sizes)),
    [n / max(n_batches) for n in n_batches],
    "^-.",
    color=GREEN,
    lw=2,
    markersize=7,
    label="Pasos/epoch (relativo)",
)

ax.set_xticks(range(len(batch_sizes)))
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.set_xlabel("Batch size")
ax.set_ylabel("Ruido / Memoria (norm.)")
ax2.set_ylabel("Pasos por epoch (norm.)", color=GREEN)
ax.set_title("Impacto del Batch Size", fontsize=10)
ax.grid(True, alpha=0.3)
lines = [l1, l2, l3]
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")

savefig("03-dataloader.png")

# ── 04 Training loop completo ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)
fig.suptitle("Training Loop PyTorch: Anatomía Completa", fontsize=13, fontweight="bold")

# izq: diagrama de flujo
ax = axes[0]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Pasos por epoch", fontsize=10)
loop_steps = [
    (0.5, 0.92, "model.train()", GREEN, 0.50, 0.10),
    (0.5, 0.78, "for batch in train_loader:", BLUE, 0.60, 0.10),
    (
        0.5,
        0.63,
        "optimizer.zero_grad()\n(limpiar gradientes acumulados)",
        ORANGE,
        0.68,
        0.12,
    ),
    (
        0.5,
        0.47,
        "outputs = model(inputs)\nloss = criterion(outputs, labels)",
        PURPLE,
        0.68,
        0.12,
    ),
    (0.5, 0.30, "loss.backward()\n(calcular gradientes)", RED, 0.68, 0.12),
    (0.5, 0.13, "optimizer.step()\n(actualizar pesos)", GREEN, 0.68, 0.12),
]
for x, y, lbl, c, w, h in loop_steps:
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02",
            facecolor=c,
            alpha=0.75,
            edgecolor="white",
            lw=2,
        )
    )
    ax.text(
        x,
        y,
        lbl,
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
    )
# flechas entre pasos
ys = [s[1] for s in loop_steps]
for i in range(len(ys) - 1):
    h_top = loop_steps[i][5]
    h_bot = loop_steps[i + 1][5]
    ax.annotate(
        "",
        xy=(0.5, ys[i + 1] + h_bot / 2),
        xytext=(0.5, ys[i] - h_top / 2),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=GRAY),
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# der: curvas de loss simulando entrenamiento PyTorch
ax = axes[1]
ax.set_facecolor(BG)
epochs = np.arange(1, 81)
np.random.seed(7)
tr = 2.5 * np.exp(-0.06 * epochs) + 0.20 + 0.03 * np.random.randn(80)
va = 2.6 * np.exp(-0.055 * epochs) + 0.28 + 0.04 * np.random.randn(80)
va = np.maximum(va, tr + 0.02)

ax.plot(epochs, tr, color=BLUE, lw=2.5, label="Train loss")
ax.plot(epochs, va, color=ORANGE, lw=2.5, label="Val loss", linestyle="--")

# best model checkpoint
best_ep = np.argmin(va) + 1
ax.axvline(
    best_ep,
    color=GREEN,
    linestyle=":",
    lw=2,
    alpha=0.8,
    label=f"Best checkpoint (ep {best_ep})",
)
ax.scatter([best_ep], [va[best_ep - 1]], s=120, color=GREEN, zorder=6)

# early stopping marker
es_ep = best_ep + 15
if es_ep < 80:
    ax.axvline(
        es_ep,
        color=RED,
        linestyle=":",
        lw=2,
        alpha=0.8,
        label=f"Early stop (ep {es_ep})",
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss durante entrenamiento PyTorch")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

savefig("04-training-loop.png")

# ── 05 Learning rate schedulers ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=BG)
fig.suptitle("Learning Rate Schedulers en PyTorch", fontsize=14, fontweight="bold")
epochs = np.arange(0, 100)


def step_lr(ep, lr0=0.1, step=25, gamma=0.5):
    return lr0 * gamma ** (ep // step)


def exp_lr(ep, lr0=0.1, gamma=0.96):
    return lr0 * gamma**ep


def cosine_lr(ep, lr0=0.1, T_max=100, lr_min=1e-4):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * ep / T_max))


def warmup_cosine(ep, lr0=0.1, warmup=10, T_max=100, lr_min=1e-4):
    if ep < warmup:
        return lr0 * ep / warmup
    return lr_min + 0.5 * (lr0 - lr_min) * (
        1 + np.cos(np.pi * (ep - warmup) / (T_max - warmup))
    )


def cyclic_lr(ep, lr_min=0.001, lr_max=0.1, T=20):
    cycle = ep % T
    return lr_min + (lr_max - lr_min) * max(0, 1 - abs(2 * cycle / T - 1))


def one_cycle(ep, lr0=0.1, T=100):
    # simplified one-cycle
    if ep < T // 2:
        return 0.001 + (lr0 - 0.001) * (ep / (T // 2))
    else:
        return lr0 * (1 - (ep - T // 2) / (T // 2)) ** 2 + 1e-5


schedulers = [
    ("StepLR\n(gamma=0.5, step=25)", step_lr, BLUE),
    ("ExponentialLR\n(gamma=0.96)", exp_lr, ORANGE),
    ("CosineAnnealingLR", cosine_lr, GREEN),
    ("WarmupCosine\n(warmup=10)", warmup_cosine, RED),
    ("CyclicLR", cyclic_lr, PURPLE),
    ("OneCycleLR", one_cycle, CYAN),
]

for ax, (name, fn, color) in zip(axes.flat, schedulers):
    ax.set_facecolor(BG)
    lr_vals = [fn(e) for e in epochs]
    ax.plot(epochs, lr_vals, color=color, lw=2.5)
    ax.fill_between(epochs, lr_vals, alpha=0.1, color=color)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.annotate(
        f"max={max(lr_vals):.3f}",
        xy=(0, max(lr_vals)),
        xytext=(5, max(lr_vals) * 0.85),
        fontsize=8,
        color=color,
    )

savefig("05-lr-schedulers.png")

# ── 06 GPU vs CPU timing ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("CPU vs GPU: Cuando Vale la Pena", fontsize=13, fontweight="bold")

# speedup simulado basado en benchmarks reales
matrix_sizes = [128, 256, 512, 1024, 2048, 4096]
# en ms (approx) — valores representativos
cpu_times = [0.3, 1.2, 8.0, 55.0, 380.0, 2800.0]
gpu_times = [0.5, 0.6, 0.8, 2.5, 12.0, 65.0]  # includes transfer overhead

ax = axes[0]
ax.set_facecolor(BG)
x_ = np.arange(len(matrix_sizes))
w_ = 0.35
ax.bar(x_ - w_ / 2, cpu_times, w_, label="CPU", color=BLUE, alpha=0.8)
ax.bar(x_ + w_ / 2, gpu_times, w_, label="GPU", color=GREEN, alpha=0.8)
ax.set_yscale("log")
ax.set_xticks(x_)
ax.set_xticklabels([f"{s}×{s}" for s in matrix_sizes], rotation=30)
ax.set_xlabel("Tamano de matriz")
ax.set_ylabel("Tiempo (ms, log scale)")
ax.set_title("matmul: CPU vs GPU (aprox.)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# speedup ratio
ax = axes[1]
ax.set_facecolor(BG)
speedup = [c / g for c, g in zip(cpu_times, gpu_times)]
colors_bar = [ORANGE if s < 2 else GREEN for s in speedup]
bars = ax.bar(
    range(len(matrix_sizes)),
    speedup,
    color=colors_bar,
    alpha=0.85,
    edgecolor="white",
    lw=1.5,
)
ax.axhline(1.0, color=RED, linestyle="--", lw=1.5, alpha=0.7, label="Sin ventaja")
for bar, s in zip(bars, speedup):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{s:.1f}x",
        ha="center",
        fontsize=9,
        color="black",
        fontweight="bold",
    )
ax.set_xticks(range(len(matrix_sizes)))
ax.set_xticklabels([f"{s}×{s}" for s in matrix_sizes], rotation=30)
ax.set_xlabel("Tamano de matriz")
ax.set_ylabel("Speedup GPU/CPU")
ax.set_title("Speedup relativo GPU vs CPU")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.text(
    0.5,
    0.92,
    "GPU es lenta para matrices pequeñas\n(overhead de transferencia)",
    transform=ax.transAxes,
    ha="center",
    fontsize=8.5,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor=ORANGE, alpha=0.8),
)

savefig("06-cpu-vs-gpu.png")

# ── 07 Arquitectura MLP en PyTorch (bloques) ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title(
    "Arquitectura MLP en PyTorch: nn.Sequential vs nn.Module",
    fontsize=13,
    fontweight="bold",
)

# columna izquierda: nn.Sequential
left_blocks = [
    ("Input\n[batch, 784]", CYAN),
    ("nn.Linear(784, 256)", BLUE),
    ("nn.BatchNorm1d(256)", PURPLE),
    ("nn.ReLU()", GREEN),
    ("nn.Dropout(0.3)", ORANGE),
    ("nn.Linear(256, 128)", BLUE),
    ("nn.BatchNorm1d(128)", PURPLE),
    ("nn.ReLU()", GREEN),
    ("nn.Dropout(0.2)", ORANGE),
    ("nn.Linear(128, 10)", RED),
    ("Output\n[batch, 10]", RED),
]

# columna derecha: nn.Module custom
right_lines = [
    ("class MLP(nn.Module):", BLUE),
    ("  def __init__(self):", BLUE),
    ("    super().__init__()", GRAY),
    ("    self.fc1 = nn.Linear(784,256)", CYAN),
    ("    self.bn1 = nn.BatchNorm1d(256)", PURPLE),
    ("    self.drop = nn.Dropout(0.3)", ORANGE),
    ("    self.fc2 = nn.Linear(256,10)", CYAN),
    ("", GRAY),
    ("  def forward(self, x):", GREEN),
    ("    x = F.relu(self.bn1(self.fc1(x)))", GREEN),
    ("    x = self.drop(x)", ORANGE),
    ("    return self.fc2(x)", RED),
    ("", GRAY),
    ("  # Ventaja: logica arbitraria,", GRAY),
    ("  # skip connections, multi-branch", GRAY),
]

# dibujar bloques izq
for i, (lbl, c) in enumerate(left_blocks):
    y_pos = 0.93 - i * 0.082
    ax.add_patch(
        FancyBboxPatch(
            (0.03, y_pos - 0.034),
            0.35,
            0.068,
            boxstyle="round,pad=0.01",
            facecolor=c,
            alpha=0.75,
            edgecolor="white",
            lw=1.5,
        )
    )
    ax.text(
        0.21,
        y_pos,
        lbl,
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
    )
    if i < len(left_blocks) - 1:
        ax.annotate(
            "",
            xy=(0.21, y_pos - 0.034 - 0.005),
            xytext=(0.21, y_pos - 0.034 - 0.001 - 0.038),
            arrowprops=dict(arrowstyle="->", lw=1.2, color=GRAY),
        )

# etiqueta columna izq
ax.text(
    0.21, 0.97, "nn.Sequential", ha="center", fontsize=10, fontweight="bold", color=BLUE
)

# dibujar codigo der
for i, (line, c) in enumerate(right_lines):
    ax.text(
        0.43,
        0.93 - i * 0.057,
        line,
        fontsize=8.5,
        color=c,
        family="monospace",
        va="top",
    )

ax.text(
    0.65,
    0.97,
    "nn.Module (custom)",
    ha="center",
    fontsize=10,
    fontweight="bold",
    color=GREEN,
)

# linea divisoria
ax.axvline(0.41, color=GRAY, linestyle="--", alpha=0.4, lw=1.5, ymin=0.02, ymax=0.96)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

savefig("07-arquitectura-mlp-pytorch.png")

# ── 08 Checkpoints y early stopping ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Checkpoints y Early Stopping", fontsize=13, fontweight="bold")

# izq: estrategia de checkpoint
ax = axes[0]
ax.set_facecolor(BG)
ax.axis("off")
ax.set_title("Estrategia de guardado", fontsize=10)
strategy = [
    (0.5, 0.88, "Checkpoint cada N epochs\n(torch.save completo)", BLUE, 0.88, 0.12),
    (0.5, 0.68, "Guardar solo state_dict\n(modelo + optimizer)", GREEN, 0.88, 0.12),
    (0.5, 0.48, "Mejor modelo segun\nval_loss (best model)", PURPLE, 0.88, 0.12),
    (0.5, 0.28, "Early stopping:\nn_iter_no_change=15", ORANGE, 0.88, 0.12),
    (0.5, 0.08, "Restaurar best model\nantes de inferencia", RED, 0.88, 0.12),
]
for x, y, lbl, c, w, h in strategy:
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02",
            facecolor=c,
            alpha=0.7,
            edgecolor="white",
            lw=2,
        )
    )
    ax.text(
        x,
        y,
        lbl,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
for i in range(len(strategy) - 1):
    y0 = strategy[i][1] - strategy[i][5] / 2
    y1 = strategy[i + 1][1] + strategy[i + 1][5] / 2
    ax.annotate(
        "",
        xy=(0.5, y1),
        xytext=(0.5, y0),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=GRAY),
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# der: val loss con best checkpoint y early stop indicados
ax = axes[1]
ax.set_facecolor(BG)
np.random.seed(12)
ep = np.arange(1, 101)
val_loss = 1.5 * np.exp(-0.04 * ep) + 0.25 + 0.015 * np.random.randn(100)
# degradacion artificial despues de ep 45
val_loss[44:] += np.linspace(0, 0.2, 56)

best_ep = np.argmin(val_loss) + 1
patience = 20
es_ep = best_ep + patience

ax.plot(ep, val_loss, color=BLUE, lw=2.5, label="Val loss")
ax.scatter(
    [best_ep],
    [val_loss[best_ep - 1]],
    s=150,
    color=GREEN,
    zorder=6,
    label=f"Best model (ep {best_ep})",
)
ax.axvline(best_ep, color=GREEN, linestyle=":", lw=2, alpha=0.7)
if es_ep <= 100:
    ax.axvline(
        es_ep,
        color=RED,
        linestyle="--",
        lw=2,
        alpha=0.8,
        label=f"Early stop (ep {es_ep}, patience={patience})",
    )
    ax.axvspan(best_ep, es_ep, alpha=0.07, color=RED, label="Zona de espera")
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Loss")
ax.set_title("Checkpoint + Early Stopping")
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

savefig("08-checkpoints.png")

# ── 09 Dashboard resumen PyTorch ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
fig.suptitle("Dashboard: Fundamentos PyTorch", fontsize=14, fontweight="bold")
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# A) LR schedulers comparados
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor(BG)
ep_ = np.arange(0, 100)
ax.plot(ep_, [cosine_lr(e) for e in ep_], color=GREEN, lw=2, label="Cosine")
ax.plot(ep_, [warmup_cosine(e) for e in ep_], color=BLUE, lw=2, label="Warmup+Cosine")
ax.plot(ep_, [one_cycle(e) for e in ep_], color=ORANGE, lw=2, label="OneCycle")
ax.set_title("LR Schedulers")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlabel("Epoch")
ax.set_ylabel("LR")

# B) Training curves con early stopping
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(BG)
ax.plot(ep, val_loss, color=BLUE, lw=2, label="Val")
ax.scatter([best_ep], [val_loss[best_ep - 1]], s=100, color=GREEN, zorder=5)
ax.axvline(best_ep, color=GREEN, linestyle=":", lw=1.5, alpha=0.7)
ax.set_title("Early Stopping")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

# C) GPU speedup
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor(BG)
colors_sp = [ORANGE if s < 2 else GREEN for s in speedup]
ax.bar(
    range(len(matrix_sizes)), speedup, color=colors_sp, alpha=0.85, edgecolor="white"
)
ax.axhline(1.0, color=RED, lw=1.5, linestyle="--", alpha=0.7)
ax.set_xticks(range(len(matrix_sizes)))
ax.set_xticklabels([str(s) for s in matrix_sizes], rotation=30, fontsize=8)
ax.set_title("GPU Speedup vs CPU")
ax.set_xlabel("Matriz N×N")
ax.set_ylabel("Speedup")
ax.grid(True, alpha=0.3, axis="y")

# D) Activaciones + autograd
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor(BG)
x_ = np.linspace(-3, 3, 200)
ax.plot(x_, np.maximum(0, x_), color=GREEN, lw=2, label="ReLU")
ax.plot(x_, 1 / (1 + np.exp(-x_)), color=BLUE, lw=2, label="Sigmoid")
ax.plot(x_, np.tanh(x_), color=ORANGE, lw=2, label="Tanh")
ax.set_title("Activaciones")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# E) Batch size impact
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor(BG)
ax.plot(range(len(batch_sizes)), noise_grad, "o-", color=RED, lw=2, label="Ruido grad.")
ax.plot(range(len(batch_sizes)), memory, "s--", color=BLUE, lw=2, label="Memoria")
ax.set_xticks(range(len(batch_sizes)))
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.set_title("Batch Size Trade-off")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# F) Checklist entrenamiento
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor(BG)
ax.axis("off")
checklist = [
    ("✓", "model.train() en train", GREEN),
    ("✓", "model.eval() en val/test", GREEN),
    ("✓", "torch.no_grad() en val", GREEN),
    ("✓", "optimizer.zero_grad()", BLUE),
    ("✓", "loss.backward()", BLUE),
    ("✓", "optimizer.step()", BLUE),
    ("✓", "Guardar best checkpoint", PURPLE),
    ("✓", "Fijar semilla aleatoria", ORANGE),
]
ax.set_title("Checklist Entrenamiento", fontsize=10)
for i, (tick, text, c) in enumerate(checklist):
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.89 - i * 0.11),
            0.96,
            0.095,
            boxstyle="round,pad=0.01",
            facecolor=c,
            alpha=0.2,
            edgecolor=c,
            lw=1,
        )
    )
    ax.text(
        0.07,
        0.89 - i * 0.11 + 0.047,
        tick,
        ha="center",
        va="center",
        fontsize=11,
        color=c,
        fontweight="bold",
    )
    ax.text(
        0.15,
        0.89 - i * 0.11 + 0.047,
        text,
        ha="left",
        va="center",
        fontsize=8.5,
        color="black",
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

savefig("09-dashboard.png")

print("✓ Todos los graficos de tema-11 generados.")
