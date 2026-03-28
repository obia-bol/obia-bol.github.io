"""Genera 9 graficos para tema-13: Redes Convolucionales (CNNs)."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings

warnings.filterwarnings("ignore")
import os

OUT = "public/ruta-aprendizaje-graficos/tema-13"
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


# ── 01 Imagen como tensor: canales RGB ───────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(14, 4), facecolor=BG)
fig.suptitle("Imagen como Tensor: Shape [C, H, W]", fontsize=13, fontweight="bold")

# Imagen sintetica 8x8
np.random.seed(7)
img_r = np.random.randint(100, 255, (8, 8))
img_g = np.random.randint(50, 150, (8, 8))
img_b = np.random.randint(0, 100, (8, 8))
img_rgb = np.stack([img_r, img_g, img_b], axis=-1).clip(0, 255).astype(np.uint8)

axes[0].imshow(img_rgb)
axes[0].set_title("Imagen RGB\nshape: [3, 8, 8]", fontsize=9, fontweight="bold")
axes[0].axis("off")

cmaps = ["Reds", "Greens", "Blues"]
titles = ["Canal R\nshape: [8, 8]", "Canal G\nshape: [8, 8]", "Canal B\nshape: [8, 8]"]
channels = [img_r, img_g, img_b]
for ax, ch, cmap, title in zip(axes[1:4], channels, cmaps, titles):
    im = ax.imshow(ch, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

# Visualizacion de valores de pixel
axes[4].set_facecolor(BG)
axes[4].axis("off")
axes[4].set_title(
    "Valores de pixel\n(canal R, primeras 4x4)", fontsize=9, fontweight="bold"
)
table_data = img_r[:4, :4].astype(str)
tbl = axes[4].table(
    cellText=table_data,
    cellLoc="center",
    loc="center",
    cellColours=plt.cm.Reds(img_r[:4, :4] / 255),
)
tbl.scale(1, 2)
tbl.set_fontsize(9)

savefig("01-imagen-tensor-rgb.png")


# ── 02 Operacion de convolucion ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig.suptitle(
    "Operacion de Convolucion: Kernel Deslizante", fontsize=13, fontweight="bold"
)

# Input 5x5
input_map = np.array(
    [
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
    ],
    dtype=float,
)

# Kernel 3x3 (detector de bordes verticales)
kernel = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ],
    dtype=float,
)

# Feature map 3x3 (stride=1, no padding)
from scipy.signal import convolve2d

feature_map = convolve2d(input_map, kernel[::-1, ::-1], mode="valid")

# Input con region resaltada
ax = axes[0]
ax.set_facecolor(BG)
ax.imshow(input_map, cmap="Blues", vmin=-0.5, vmax=1.5)
for i in range(5):
    for j in range(5):
        ax.text(
            j,
            i,
            f"{int(input_map[i,j])}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white" if input_map[i, j] > 0.5 else "black",
        )
# Resaltar region [0:3, 0:3]
rect = plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor=ORANGE, lw=3)
ax.add_patch(rect)
ax.set_title("Input (5×5)\nRegion activa marcada", fontsize=10, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])

# Kernel
ax = axes[1]
ax.set_facecolor(BG)
ax.imshow(kernel, cmap="RdBu", vmin=-2, vmax=2)
for i in range(3):
    for j in range(3):
        ax.text(
            j,
            i,
            f"{int(kernel[i,j])}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white" if abs(kernel[i, j]) > 0.5 else "black",
        )
ax.set_title("Kernel 3×3\n(detector bordes verticales)", fontsize=10, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])

# Feature map
ax = axes[2]
ax.set_facecolor(BG)
vmax = np.abs(feature_map).max()
im = ax.imshow(feature_map, cmap="RdBu", vmin=-vmax, vmax=vmax)
for i in range(3):
    for j in range(3):
        ax.text(
            j,
            i,
            f"{feature_map[i,j]:.0f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title(
    "Feature Map (3×3)\nFormula: Σ(input · kernel) + bias",
    fontsize=10,
    fontweight="bold",
)
ax.set_xticks([])
ax.set_yticks([])

savefig("02-convolucion-kernel.png")


# ── 03 Efectos de distintos filtros ──────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor=BG)
fig.suptitle(
    "Efectos de Distintos Filtros Convolucionales", fontsize=13, fontweight="bold"
)

# Imagen sintetica con estructura (circulo + bordes)
from matplotlib.patches import Circle

img_base = np.zeros((32, 32))
for i in range(32):
    for j in range(32):
        if (i - 16) ** 2 + (j - 16) ** 2 < 100:
            img_base[i, j] = 1.0
# Agregar ruido leve y gradiente
img_base += np.random.randn(32, 32) * 0.05
img_base[8:12, :] = 0.7
img_base[:, 8:12] = 0.7
img_base = np.clip(img_base, 0, 1)

kernels_info = [
    (
        "Identidad\n(sin cambio)",
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    ),
    ("Blur\n(suavizado)", np.ones((3, 3)) / 9),
    (
        "Nitidez\n(sharpen)",
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    ),
    ("Bordes H\n(Sobel)", np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)),
    ("Bordes V\n(Sobel)", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)),
    ("Emboss\n(relieve)", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float)),
    (
        "Laplacian\n(bordes todos)",
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float),
    ),
    (
        "Diagonal\n(detector)",
        np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=float),
    ),
]

for ax, (title, k) in zip(axes.flatten(), kernels_info):
    result = convolve2d(img_base, k, mode="same")
    result = (
        np.clip(result, 0, 1)
        if result.min() >= 0
        else (result - result.min()) / (result.max() - result.min() + 1e-8)
    )
    ax.imshow(result, cmap="gray")
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.axis("off")

# Mostrar imagen original en primera celda
axes[0, 0].imshow(img_base, cmap="gray")
axes[0, 0].set_title("Original", fontsize=8.5, fontweight="bold")

savefig("03-efectos-filtros.png")


# ── 04 Pooling: Max vs Average ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor=BG)
fig.suptitle(
    "Pooling: Reduccion Espacial de Feature Maps", fontsize=13, fontweight="bold"
)

feature = np.array(
    [
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [1, 2, 3, 4],
        [5, 7, 2, 1],
    ],
    dtype=float,
)

max_pool = np.array([[6, 4], [7, 4]], dtype=float)
avg_pool = np.array([[15 / 4, 9 / 4], [15 / 4, 10 / 4]], dtype=float)


def show_pool(ax, data, title, highlight=None, cmap="Blues"):
    ax.set_facecolor(BG)
    n = data.shape[0]
    ax.imshow(data, cmap=cmap, vmin=0, vmax=data.max() * 1.1)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{data[i,j]:.1f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if data[i, j] > data.max() * 0.5 else "black",
            )
    if highlight:
        for r1, c1, r2, c2 in highlight:
            rect = plt.Rectangle(
                (c1 - 0.5, r1 - 0.5),
                c2 - c1,
                r2 - r1,
                fill=False,
                edgecolor=ORANGE,
                lw=3,
            )
            ax.add_patch(rect)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


# Fila 1: Max Pooling
show_pool(
    axes[0, 0],
    feature,
    "Feature Map (4×4)",
    highlight=[(0, 0, 2, 2), (0, 2, 2, 4), (2, 0, 4, 2), (2, 2, 4, 4)],
)
show_pool(
    axes[0, 1],
    max_pool,
    "Max Pool (2×2, stride=2)\nToma el MAXIMO de cada region",
    cmap="Oranges",
)
axes[0, 2].set_facecolor(BG)
axes[0, 2].axis("off")
axes[0, 2].text(
    0.5, 0.65, "Max Pooling", fontsize=12, ha="center", fontweight="bold", color=ORANGE
)
axes[0, 2].text(
    0.5,
    0.45,
    "• Conserva la activacion mas fuerte\n• Robusto a desplazamientos\n• Mas usado en CNNs clasicas\n• Adecuado para detectar presencia\n  de features",
    fontsize=9,
    ha="center",
    va="center",
)

# Fila 2: Average Pooling
show_pool(
    axes[1, 0],
    feature,
    "Feature Map (4×4)",
    highlight=[(0, 0, 2, 2), (0, 2, 2, 4), (2, 0, 4, 2), (2, 2, 4, 4)],
)
show_pool(
    axes[1, 1],
    avg_pool,
    "Avg Pool (2×2, stride=2)\nToma el PROMEDIO de cada region",
    cmap="Greens",
)
axes[1, 2].set_facecolor(BG)
axes[1, 2].axis("off")
axes[1, 2].text(
    0.5,
    0.65,
    "Average Pooling",
    fontsize=12,
    ha="center",
    fontweight="bold",
    color=GREEN,
)
axes[1, 2].text(
    0.5,
    0.45,
    "• Promedia todas las activaciones\n• Menos agresivo\n• Usado en capas finales (GAP)\n• Global Avg Pool elimina dims\n  espaciales antes del clasificador",
    fontsize=9,
    ha="center",
    va="center",
)

savefig("04-pooling-max-avg.png")


# ── 05 Arquitectura CNN completa ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
fig.suptitle(
    "Arquitectura CNN: De Pixeles a Prediccion", fontsize=13, fontweight="bold"
)

blocks = [
    # (x, y, w, h, color, label1, label2)
    (0.3, 1.5, 1.2, 3.0, BLUE, "Input", "32×32×3"),
    (2.0, 1.8, 1.0, 2.4, CYAN, "Conv1\n+ReLU", "30×30×32"),
    (3.4, 2.0, 0.8, 2.0, CYAN, "Conv2\n+ReLU", "28×28×32"),
    (4.6, 2.3, 0.7, 1.4, PURPLE, "MaxPool", "14×14×32"),
    (5.8, 2.1, 1.0, 1.8, CYAN, "Conv3\n+ReLU", "12×12×64"),
    (7.2, 2.3, 1.0, 1.4, CYAN, "Conv4\n+ReLU", "10×10×64"),
    (8.7, 2.5, 0.7, 1.0, PURPLE, "MaxPool", "5×5×64"),
    (9.9, 2.6, 0.6, 0.8, GRAY, "Flatten", "1600"),
    (11.0, 2.6, 0.8, 0.8, ORANGE, "FC\n+ReLU", "256"),
    (12.3, 2.6, 0.8, 0.8, ORANGE, "Dropout", "p=0.5"),
    (13.5, 2.6, 0.8, 0.8, RED, "FC\nSoftmax", "10 clases"),
]

for x, y, w, h, c, lab1, lab2 in blocks:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor=c,
            alpha=0.75,
            edgecolor="white",
            lw=1.5,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2 + 0.15,
        lab1,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.25,
        lab2,
        ha="center",
        va="center",
        fontsize=6.5,
        color="white",
        alpha=0.9,
    )

# Flechas de conexion
arrow_xs = [
    (1.5, 2.0),
    (3.0, 3.4),
    (4.2, 4.6),
    (5.3, 5.8),
    (6.8, 7.2),
    (8.2, 8.7),
    (9.4, 9.9),
    (10.5, 11.0),
    (11.8, 12.3),
    (13.1, 13.5),
]
for x1, x2 in arrow_xs:
    ax.annotate(
        "",
        xy=(x2, 3.0),
        xytext=(x1, 3.0),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5),
    )

# Leyenda
legend_items = [
    mpatches.Patch(color=BLUE, label="Input"),
    mpatches.Patch(color=CYAN, label="Conv + ReLU"),
    mpatches.Patch(color=PURPLE, label="Pooling"),
    mpatches.Patch(color=GRAY, label="Flatten"),
    mpatches.Patch(color=ORANGE, label="Fully Connected"),
    mpatches.Patch(color=RED, label="Salida"),
]
ax.legend(
    handles=legend_items,
    loc="lower center",
    ncol=6,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.05),
)

# Anotacion de bloques
ax.text(
    3.2,
    5.2,
    "Bloque Convolucional 1\n(extrae features basicas)",
    fontsize=8,
    ha="center",
    color=CYAN,
    style="italic",
)
ax.add_patch(
    plt.Rectangle((1.9, 1.7), 3.4, 3.7, fill=False, edgecolor=CYAN, lw=1.5, ls="--")
)

ax.text(
    7.5,
    5.2,
    "Bloque Convolucional 2\n(features mas complejas)",
    fontsize=8,
    ha="center",
    color=CYAN,
    style="italic",
)
ax.add_patch(
    plt.Rectangle((5.7, 1.9), 3.7, 3.5, fill=False, edgecolor=CYAN, lw=1.5, ls="--")
)

ax.text(
    12.0,
    5.2,
    "Clasificador\n(capas densas)",
    fontsize=8,
    ha="center",
    color=ORANGE,
    style="italic",
)
ax.add_patch(
    plt.Rectangle((9.8, 2.4), 4.7, 2.1, fill=False, edgecolor=ORANGE, lw=1.5, ls="--")
)

savefig("05-arquitectura-cnn.png")


# ── 06 Transfer Learning: feature extraction vs fine-tuning ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 7), facecolor=BG)
fig.suptitle(
    "Transfer Learning: Estrategias de Reutilizacion de Pesos",
    fontsize=13,
    fontweight="bold",
)


def draw_network(ax, title, frozen_layers, trainable_layers, note):
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

    layer_configs = [
        (5.0, 10.5, "Conv1-3\nFeatures basicas\n(bordes, texturas)", 2.5, 1.0),
        (5.0, 8.5, "Conv4-6\nFeatures medias\n(formas, patrones)", 2.5, 1.0),
        (5.0, 6.5, "Conv7-9\nFeatures altas\n(partes objetos)", 2.5, 1.0),
        (5.0, 4.5, "FC / Head\nClasificacion", 2.5, 1.0),
    ]

    all_layers = frozen_layers + trainable_layers
    colors_map = {}
    for l in frozen_layers:
        colors_map[l] = ("#aaaaaa", "Congelado (frozen)")
    for l in trainable_layers:
        colors_map[l] = (GREEN, "Entrenable")

    for i, (cx, cy, label, w, h) in enumerate(layer_configs):
        color, _ = colors_map.get(i, (BLUE, ""))
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0.05",
                facecolor=color,
                alpha=0.8,
                edgecolor="white",
                lw=1.5,
            )
        )
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white",
        )
        if i in frozen_layers:
            ax.text(cx + 1.5, cy, "🔒", fontsize=10, ha="center", va="center")

    ax.text(
        5.0,
        1.8,
        note,
        ha="center",
        va="center",
        fontsize=8,
        color=GRAY,
        style="italic",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor=GRAY, lw=1),
    )


# Entrenamiento desde cero
draw_network(
    axes[0],
    "Desde Cero",
    frozen_layers=[],
    trainable_layers=[0, 1, 2, 3],
    note="Todos los pesos son aleatorios.\nNecesita mucho dato y tiempo.\nRara vez recomendado.",
)

# Feature extraction
draw_network(
    axes[1],
    "Feature Extraction\n(congelar todo excepto head)",
    frozen_layers=[0, 1, 2],
    trainable_layers=[3],
    note="Solo entrena el clasificador final.\nRapido, pocos datos necesarios.\nBueno cuando dominio es similar.",
)

# Fine-tuning
draw_network(
    axes[2],
    "Fine-Tuning\n(descongelar capas altas)",
    frozen_layers=[0, 1],
    trainable_layers=[2, 3],
    note="Entrena capas altas + head.\nMejor accuracy que feature ext.\nRequiere algo mas de datos y LR bajo.",
)

# Leyenda
legend_items = [
    mpatches.Patch(color="#aaaaaa", label="Congelado (pesos de ImageNet)"),
    mpatches.Patch(color=GREEN, label="Entrenable"),
]
fig.legend(
    handles=legend_items,
    loc="lower center",
    ncol=2,
    fontsize=9,
    bbox_to_anchor=(0.5, 0.01),
)

savefig("06-transfer-learning.png")


# ── 07 Image Augmentation ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(14, 6), facecolor=BG)
fig.suptitle(
    "Image Augmentation: Aumentar la Variedad de Datos", fontsize=13, fontweight="bold"
)

# Imagen base sintetica (gradiente + circulo)
base = np.zeros((64, 64, 3), dtype=np.uint8)
base[:, :, 0] = np.linspace(60, 200, 64)[np.newaxis, :] * np.ones((64, 1))
base[:, :, 1] = np.linspace(80, 160, 64)[:, np.newaxis] * np.ones((1, 64))
base[:, :, 2] = 100
for i in range(64):
    for j in range(64):
        if (i - 32) ** 2 + (j - 32) ** 2 < 300:
            base[i, j] = [220, 80, 60]
base = np.clip(base, 0, 255).astype(np.uint8)


def flip_h(img):
    return img[:, ::-1]


def flip_v(img):
    return img[::-1, :]


def rotate90(img):
    return np.rot90(img)


def crop_center(img, frac=0.75):
    h, w = img.shape[:2]
    s = int(min(h, w) * frac)
    r = (h - s) // 2
    c = (w - s) // 2
    cropped = img[r : r + s, c : c + s]
    from PIL import Image

    return np.array(Image.fromarray(cropped).resize((h, w)))


def brightness(img, factor=1.5):
    return np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)


def add_noise(img, sigma=20):
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def cutout(img, size=20):
    r = np.copy(img)
    h, w = img.shape[:2]
    y, x = np.random.randint(size, h - size), np.random.randint(size, w - size)
    r[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2] = 128
    return r


def color_jitter(img):
    hsv = img.astype(float)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] * np.random.uniform(0.7, 1.3), 0, 255)
    return hsv.astype(np.uint8)


def blur(img):
    k = np.ones((3, 3)) / 9
    r = np.stack(
        [convolve2d(img[:, :, c].astype(float), k, mode="same") for c in range(3)],
        axis=-1,
    )
    return np.clip(r, 0, 255).astype(np.uint8)


try:
    augmentations = [
        ("Original", base),
        ("Flip Horizontal", flip_h(base)),
        ("Flip Vertical", flip_v(base)),
        ("Rotacion 90°", rotate90(base)),
        ("Brillo ×1.5", brightness(base, 1.5)),
        ("Oscurecido ×0.5", brightness(base, 0.5)),
        ("Ruido Gaussiano", add_noise(base, 25)),
        ("Cutout", cutout(base, 15)),
        ("Color Jitter", color_jitter(base)),
        ("Blur", blur(base)),
    ]
except Exception:
    augmentations = [(f"Augment {i}", base) for i in range(10)]

for ax, (title, img) in zip(axes.flatten(), augmentations):
    ax.imshow(img)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")

savefig("07-image-augmentation.png")


# ── 08 Desde cero vs transfer learning: curvas de entrenamiento ──────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)
fig.suptitle(
    "Desde Cero vs Transfer Learning: Curvas de Entrenamiento",
    fontsize=13,
    fontweight="bold",
)

epochs = np.arange(1, 51)

# Desde cero: convergencia lenta, mayor gap
tr_scratch = 0.92 * (1 - np.exp(-0.05 * epochs)) + np.random.randn(50) * 0.012
vl_scratch = 0.62 * (1 - np.exp(-0.03 * epochs)) + np.random.randn(50) * 0.015
vl_scratch = np.clip(vl_scratch, 0, 1)

# Transfer learning: convergencia rapida, mejor val acc
tr_transfer = 0.97 * (1 - np.exp(-0.12 * epochs)) + np.random.randn(50) * 0.007
vl_transfer = 0.92 * (1 - np.exp(-0.11 * epochs)) + np.random.randn(50) * 0.009

# Loss
ax = axes[0]
ax.set_facecolor(BG)
loss_s_tr = 1.5 * np.exp(-0.04 * epochs) + 0.1 + np.random.randn(50) * 0.02
loss_s_vl = (
    1.5 * np.exp(-0.025 * epochs)
    + 0.4
    + 0.005 * (epochs - 30) * (epochs > 30)
    + np.random.randn(50) * 0.025
)
loss_t_tr = 1.5 * np.exp(-0.1 * epochs) + 0.04 + np.random.randn(50) * 0.01
loss_t_vl = 1.5 * np.exp(-0.095 * epochs) + 0.06 + np.random.randn(50) * 0.012
ax.plot(epochs, loss_s_tr, color=ORANGE, lw=2, label="Scratch - Train")
ax.plot(epochs, loss_s_vl, color=ORANGE, lw=2, ls="--", label="Scratch - Val")
ax.plot(epochs, loss_t_tr, color=BLUE, lw=2, label="Transfer - Train")
ax.plot(epochs, loss_t_vl, color=BLUE, lw=2, ls="--", label="Transfer - Val")
ax.set_title("Curvas de Perdida (Loss)", fontsize=11, fontweight="bold")
ax.set_xlabel("Epoca")
ax.set_ylabel("Cross-Entropy Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Accuracy
ax = axes[1]
ax.set_facecolor(BG)
ax.plot(epochs, tr_scratch, color=ORANGE, lw=2, label="Scratch - Train")
ax.plot(epochs, vl_scratch, color=ORANGE, lw=2, ls="--", label="Scratch - Val")
ax.plot(epochs, tr_transfer, color=BLUE, lw=2, label="Transfer - Train")
ax.plot(epochs, vl_transfer, color=BLUE, lw=2, ls="--", label="Transfer - Val")
ax.set_title("Curvas de Accuracy", fontsize=11, fontweight="bold")
ax.set_xlabel("Epoca")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Anotacion
ax.annotate(
    f"Transfer: {vl_transfer[-1]:.2f}",
    xy=(50, vl_transfer[-1]),
    xytext=(38, vl_transfer[-1] + 0.07),
    arrowprops=dict(arrowstyle="->", color=BLUE),
    fontsize=9,
    color=BLUE,
)
ax.annotate(
    f"Scratch: {vl_scratch[-1]:.2f}",
    xy=(50, vl_scratch[-1]),
    xytext=(35, vl_scratch[-1] - 0.1),
    arrowprops=dict(arrowstyle="->", color=ORANGE),
    fontsize=9,
    color=ORANGE,
)

savefig("08-scratch-vs-transfer.png")


# ── 09 Dashboard resumen CNN ──────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10), facecolor=BG)
fig.suptitle("Dashboard: Redes Convolucionales (CNNs)", fontsize=14, fontweight="bold")
gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

ax_arch = fig.add_subplot(gs[0, :2])
ax_trans = fig.add_subplot(gs[0, 2:])
ax_aug = fig.add_subplot(gs[1, :2])
ax_tips = fig.add_subplot(gs[1, 2:])
ax_tbl = fig.add_subplot(gs[2, :])

for ax in [ax_arch, ax_trans, ax_aug, ax_tips, ax_tbl]:
    ax.set_facecolor(BG)

# Mini: numero de parametros por arquitectura
archs = ["LeNet-5", "AlexNet", "VGG-16", "ResNet-50", "EfficientNet-B0", "ConvNeXt-T"]
params = [0.06, 60, 138, 25.6, 5.3, 28]
colors_a = [CYAN, BLUE, PURPLE, GREEN, ORANGE, RED]
bars = ax_arch.bar(archs, params, color=colors_a, alpha=0.8, edgecolor="white")
ax_arch.set_title(
    "Parametros por Arquitectura (millones)", fontsize=9, fontweight="bold"
)
ax_arch.set_ylabel("Millones de parametros")
ax_arch.set_yscale("log")
ax_arch.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, params):
    ax_arch.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.1,
        f"{val}M",
        ha="center",
        fontsize=7.5,
        fontweight="bold",
    )
ax_arch.grid(True, alpha=0.3, axis="y")

# Mini: accuracy en ImageNet vs año
years = [1998, 2012, 2014, 2015, 2017, 2020, 2022]
top1_acc = [60, 63.3, 74.9, 76.1, 82.9, 84.3, 87.8]
arch_names = ["LeNet", "AlexNet", "VGG", "ResNet", "SENet", "EfficientNet", "ConvNeXt"]
ax_trans.plot(years, top1_acc, "o-", color=BLUE, lw=2.5, ms=7)
for y, a, n in zip(years, top1_acc, arch_names):
    ax_trans.annotate(n, xy=(y, a), xytext=(y - 0.5, a + 1.2), fontsize=7, rotation=30)
ax_trans.axhline(96.5, color=RED, ls="--", alpha=0.5, label="Human ~96.5%")
ax_trans.set_title("Progreso en ImageNet Top-1 Accuracy", fontsize=9, fontweight="bold")
ax_trans.set_xlabel("Año")
ax_trans.set_ylabel("Top-1 Accuracy (%)")
ax_trans.set_ylim(55, 100)
ax_trans.legend(fontsize=8)
ax_trans.grid(True, alpha=0.3)

# Mini: impacto de augmentation
aug_types = ["Sin Aug.", "Flip H.", "+Crop", "+Color", "+Cutout", "+Mixup"]
val_acc = [0.72, 0.77, 0.80, 0.83, 0.85, 0.88]
ax_aug.bar(
    aug_types,
    val_acc,
    color=[ORANGE, CYAN, BLUE, GREEN, PURPLE, RED],
    alpha=0.8,
    edgecolor="white",
)
ax_aug.set_title(
    "Impacto Acumulado de Augmentation (CIFAR-10)", fontsize=9, fontweight="bold"
)
ax_aug.set_ylabel("Val Accuracy")
ax_aug.set_ylim(0.6, 0.95)
for i, v in enumerate(val_acc):
    ax_aug.text(i, v + 0.005, f"{v:.0%}", ha="center", fontsize=8, fontweight="bold")
ax_aug.tick_params(axis="x", rotation=20)
ax_aug.grid(True, alpha=0.3, axis="y")

# Tips y errores comunes
ax_tips.axis("off")
ax_tips.set_title("Checklist CNN en Competencias", fontsize=9, fontweight="bold")
tips = [
    ("✓", GREEN, "Usa transfer learning (ResNet/EfficientNet)"),
    ("✓", GREEN, "Augmentation: flip + crop + color jitter"),
    ("✓", GREEN, "BatchNorm despues de cada Conv"),
    ("✓", GREEN, "Global Average Pooling antes del clasificador"),
    ("✗", RED, "Entrenar desde cero con pocos datos"),
    ("✗", RED, "Olvidar model.eval() en inferencia"),
    ("✗", RED, "Normalizar con stats incorrectas (usar ImageNet)"),
    ("✗", RED, "Batch size muy pequeno con BatchNorm"),
]
for i, (icon, c, text) in enumerate(tips):
    y = 0.92 - i * 0.12
    ax_tips.text(
        0.04,
        y,
        icon,
        transform=ax_tips.transAxes,
        fontsize=12,
        va="center",
        color=c,
        fontweight="bold",
    )
    ax_tips.text(
        0.12,
        y,
        text,
        transform=ax_tips.transAxes,
        fontsize=8.5,
        va="center",
        color="black",
    )

# Tabla resumen
ax_tbl.axis("off")
headers = ["Componente", "Funcion", "Hiperparametros clave", "Codigo PyTorch"]
rows = [
    [
        "nn.Conv2d",
        "Extrae features espaciales",
        "in_ch, out_ch, kernel_size, stride, padding",
        "nn.Conv2d(3, 32, 3, padding=1)",
    ],
    ["nn.BatchNorm2d", "Estabiliza activaciones", "num_features", "nn.BatchNorm2d(32)"],
    [
        "nn.ReLU",
        "No linealidad",
        "inplace=True (ahorra memoria)",
        "nn.ReLU(inplace=True)",
    ],
    [
        "nn.MaxPool2d",
        "Reduce dimensiones espaciales",
        "kernel_size=2, stride=2",
        "nn.MaxPool2d(2)",
    ],
    [
        "nn.AdaptiveAvgPool2d",
        "GAP: cualquier input → output",
        "output_size=(1,1)",
        "nn.AdaptiveAvgPool2d((1,1))",
    ],
    [
        "torchvision ResNet",
        "Backbone preentrenado",
        "pretrained=True, num_classes",
        "models.resnet50(pretrained=True)",
    ],
    [
        "transforms.v2",
        "Augmentation",
        "RandomHorizontalFlip, RandomCrop",
        "v2.RandomHorizontalFlip(p=0.5)",
    ],
]
table = ax_tbl.table(
    cellText=rows,
    colLabels=headers,
    cellLoc="left",
    loc="center",
    colWidths=[0.15, 0.22, 0.30, 0.33],
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
