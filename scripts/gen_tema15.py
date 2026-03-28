"""
gen_tema15.py — Genera 9 gráficos para el tema 15: Embeddings y Transformers
Salida: public/ruta-aprendizaje-graficos/tema-15/
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

OUT = "public/ruta-aprendizaje-graficos/tema-15"
os.makedirs(OUT, exist_ok=True)

AZUL = "#2563EB"
VERDE = "#16A34A"
ROJO = "#DC2626"
NARANJA = "#EA580C"
MORADO = "#7C3AED"
GRIS = "#6B7280"
AMARILLO = "#D97706"
CIAN = "#0891B2"
ROSA = "#DB2777"
FONDO = "#F8FAFC"
DARK = "#1E293B"


def savefig(name):
    plt.savefig(f"{OUT}/{name}", dpi=130, bbox_inches="tight", facecolor=FONDO)
    plt.close()
    print(f"  ok {name}")


# ─── 01. Espacio de embeddings: analogías geométricas ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=FONDO)
fig.suptitle(
    "Word Embeddings: Geometría del Espacio Vectorial",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

# Panel izquierdo: analogía rey-reina, hombre-mujer en 2D PCA
ax = axes[0]
ax.set_facecolor(FONDO)
ax.set_title("Analogías en el espacio (proyección 2D)", fontweight="bold", color=DARK)

puntos = {
    "rey": (0.85, 0.80),
    "reina": (0.85, 0.20),
    "hombre": (0.20, 0.80),
    "mujer": (0.20, 0.20),
    "principe": (0.65, 0.65),
    "princesa": (0.65, 0.35),
    "actor": (0.45, 0.75),
    "actriz": (0.45, 0.25),
}
colores_p = {
    "rey": AZUL,
    "reina": ROSA,
    "hombre": AZUL,
    "mujer": ROSA,
    "principe": MORADO,
    "princesa": MORADO,
    "actor": VERDE,
    "actriz": VERDE,
}

# Flechas de relación género
for a, b in [
    ("rey", "reina"),
    ("hombre", "mujer"),
    ("principe", "princesa"),
    ("actor", "actriz"),
]:
    ax.annotate(
        "",
        xy=puntos[b],
        xytext=puntos[a],
        arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.5, linestyle="dashed"),
    )

# Flechas de analogía
for a, b in [("hombre", "rey"), ("mujer", "reina")]:
    ax.annotate(
        "",
        xy=puntos[b],
        xytext=puntos[a],
        arrowprops=dict(arrowstyle="->", color=NARANJA, lw=2, alpha=0.7),
    )

for nombre, (x, y) in puntos.items():
    ax.scatter(
        x,
        y,
        s=200,
        color=colores_p[nombre],
        zorder=5,
        edgecolors="white",
        linewidth=1.5,
    )
    ax.text(
        x + 0.025,
        y + 0.03,
        nombre,
        fontsize=9.5,
        color=colores_p[nombre],
        fontweight="bold",
    )

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Dimensión 1 (Realeza →)")
ax.set_ylabel("Dimensión 2 (Masculino →)")
ax.legend(
    handles=[
        mpatches.Patch(color=GRIS, label="Relación de género"),
        mpatches.Patch(color=NARANJA, label="Analogía: es a"),
    ],
    fontsize=8,
    loc="upper left",
)
ax.grid(True, alpha=0.2)

# Panel derecho: similaridad coseno entre palabras
ax2 = axes[1]
ax2.set_facecolor(FONDO)
ax2.set_title("Similaridad coseno entre palabras", fontweight="bold", color=DARK)

palabras = ["rey", "reina", "hombre", "mujer", "gato", "perro", "auto", "avion"]
# Similaridades simuladas (inspiradas en valores reales de Word2Vec)
sim = np.array(
    [
        [1.00, 0.72, 0.68, 0.54, 0.18, 0.15, 0.10, 0.08],
        [0.72, 1.00, 0.55, 0.71, 0.20, 0.17, 0.12, 0.09],
        [0.68, 0.55, 1.00, 0.76, 0.22, 0.20, 0.18, 0.14],
        [0.54, 0.71, 0.76, 1.00, 0.25, 0.23, 0.15, 0.12],
        [0.18, 0.20, 0.22, 0.25, 1.00, 0.82, 0.12, 0.10],
        [0.15, 0.17, 0.20, 0.23, 0.82, 1.00, 0.15, 0.12],
        [0.10, 0.12, 0.18, 0.15, 0.12, 0.15, 1.00, 0.74],
        [0.08, 0.09, 0.14, 0.12, 0.10, 0.12, 0.74, 1.00],
    ]
)

im = ax2.imshow(sim, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(palabras)))
ax2.set_xticklabels(palabras, rotation=45, ha="right", fontsize=9)
ax2.set_yticks(range(len(palabras)))
ax2.set_yticklabels(palabras, fontsize=9)
for i in range(len(palabras)):
    for j in range(len(palabras)):
        ax2.text(
            j,
            i,
            f"{sim[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if sim[i, j] > 0.6 else DARK,
        )
plt.colorbar(im, ax=ax2, shrink=0.85, label="Similaridad coseno")

plt.tight_layout()
savefig("01-espacio-embeddings.png")


# ─── 02. Word2Vec: CBOW vs Skip-gram ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=FONDO)
fig.suptitle(
    "Word2Vec: Dos Arquitecturas de Entrenamiento",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)


def dibujar_red(
    ax, titulo, color_entrada, color_salida, etiq_entrada, etiq_salida, descripcion
):
    ax.set_facecolor(FONDO)
    ax.axis("off")
    ax.set_title(titulo, fontweight="bold", color=DARK, pad=8)

    # Capa de entrada
    n_in = len(etiq_entrada)
    for i, etiq in enumerate(etiq_entrada):
        y = 0.8 - i * (0.6 / max(n_in - 1, 1))
        circ = plt.Circle((0.18, y), 0.05, color=color_entrada, zorder=5)
        ax.add_patch(circ)
        ax.text(
            0.05,
            y,
            etiq,
            ha="right",
            va="center",
            fontsize=8.5,
            color=DARK,
            style="italic",
        )

    # Capa oculta (embedding)
    for i in range(3):
        y = 0.7 - i * 0.2
        circ = plt.Circle((0.5, y), 0.04, color=GRIS, zorder=5, alpha=0.6)
        ax.add_patch(circ)

    ax.text(
        0.5,
        0.14,
        "Embedding\n(pesos W)",
        ha="center",
        fontsize=8,
        color=GRIS,
        style="italic",
    )

    # Capa de salida
    n_out = len(etiq_salida)
    for i, etiq in enumerate(etiq_salida):
        y = 0.8 - i * (0.6 / max(n_out - 1, 1))
        circ = plt.Circle((0.82, y), 0.05, color=color_salida, zorder=5)
        ax.add_patch(circ)
        ax.text(
            0.95,
            y,
            etiq,
            ha="left",
            va="center",
            fontsize=8.5,
            color=DARK,
            style="italic",
        )

    # Conexiones entrada→oculta
    for i in range(n_in):
        y_in = 0.8 - i * (0.6 / max(n_in - 1, 1))
        for j in range(3):
            y_hid = 0.7 - j * 0.2
            ax.plot(
                [0.23, 0.46], [y_in, y_hid], color=color_entrada, alpha=0.25, lw=0.9
            )

    # Conexiones oculta→salida
    for j in range(3):
        y_hid = 0.7 - j * 0.2
        for k in range(n_out):
            y_out = 0.8 - k * (0.6 / max(n_out - 1, 1))
            ax.plot(
                [0.54, 0.77], [y_hid, y_out], color=color_salida, alpha=0.25, lw=0.9
            )

    # Labels de capas
    ax.text(
        0.18,
        0.95,
        "Entrada",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=color_entrada,
    )
    ax.text(
        0.50, 0.95, "Oculta", ha="center", fontsize=9, fontweight="bold", color=GRIS
    )
    ax.text(
        0.82,
        0.95,
        "Salida",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=color_salida,
    )
    ax.text(
        0.5,
        0.02,
        descripcion,
        ha="center",
        fontsize=8,
        color=DARK,
        style="italic",
        wrap=True,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)


dibujar_red(
    axes[0],
    titulo="CBOW: Contexto → Palabra central",
    color_entrada=VERDE,
    color_salida=AZUL,
    etiq_entrada=['"el"', '"come"', '"rapido"', '"hoy"'],
    etiq_salida=['"gato"'],
    descripcion="Predice la palabra central a partir del contexto",
)

dibujar_red(
    axes[1],
    titulo="Skip-gram: Palabra → Contexto",
    color_entrada=AZUL,
    color_salida=VERDE,
    etiq_entrada=['"gato"'],
    etiq_salida=['"el"', '"come"', '"rapido"', '"hoy"'],
    descripcion="Predice palabras de contexto a partir de la central\n(mejor para palabras raras)",
)

plt.tight_layout()
savefig("02-word2vec-arquitecturas.png")


# ─── 03. Mecanismo de atención: Q, K, V ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=FONDO)
fig.suptitle(
    "Mecanismo de Atención: Query, Key, Value",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

# Panel izquierdo: heatmap de atención en una oración
ax = axes[0]
ax.set_facecolor(FONDO)

tokens_frase = ["El", "banco", "del", "rio", "estaba", "seco"]
n = len(tokens_frase)
# Simular pesos de atención (la palabra "banco" atiende a "rio" y "seco" fuerte)
attn = np.array(
    [
        [0.70, 0.10, 0.08, 0.05, 0.04, 0.03],  # El
        [0.08, 0.35, 0.10, 0.28, 0.10, 0.09],  # banco (atiende mucho a rio)
        [0.05, 0.08, 0.50, 0.15, 0.12, 0.10],  # del
        [0.04, 0.20, 0.12, 0.40, 0.14, 0.10],  # rio
        [0.04, 0.08, 0.10, 0.12, 0.48, 0.18],  # estaba
        [0.04, 0.10, 0.08, 0.12, 0.18, 0.48],  # seco
    ]
)
attn = attn / attn.sum(axis=1, keepdims=True)

im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0, vmax=0.7)
ax.set_xticks(range(n))
ax.set_xticklabels(tokens_frase, fontsize=10)
ax.set_yticks(range(n))
ax.set_yticklabels(tokens_frase, fontsize=10)
ax.set_xlabel("Palabras clave (Key)")
ax.set_ylabel("Palabras consulta (Query)")
ax.set_title(
    "Heatmap de atención: 'banco' desambiguado por 'rio'",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
for i in range(n):
    for j in range(n):
        ax.text(
            j,
            i,
            f"{attn[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if attn[i, j] > 0.35 else DARK,
        )
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel derecho: diagrama Q, K, V
ax2 = axes[1]
ax2.set_facecolor(FONDO)
ax2.axis("off")
ax2.set_title(
    "Cálculo de Atención: Softmax(QK^T / sqrt(d_k)) V",
    fontsize=9.5,
    fontweight="bold",
    color=DARK,
)

# Boxes para el diagrama
cajas = [
    (0.15, 0.72, 0.22, 0.12, "Entrada X\n[seq, d_model]", AZUL, "#EFF6FF"),
    (0.15, 0.52, 0.22, 0.10, "Query Q = XW_Q", VERDE, "#F0FDF4"),
    (0.15, 0.37, 0.22, 0.10, "Key   K = XW_K", NARANJA, "#FFF7ED"),
    (0.15, 0.22, 0.22, 0.10, "Value V = XW_V", MORADO, "#FDF4FF"),
    (0.55, 0.52, 0.28, 0.10, "Scores = QK^T / sqrt(d_k)", ROJO, "#FEF2F2"),
    (0.55, 0.37, 0.28, 0.10, "Weights = Softmax(Scores)", VERDE, "#F0FDF4"),
    (0.55, 0.22, 0.28, 0.10, "Output = Weights x V", AZUL, "#EFF6FF"),
]
for x, y, w, h, txt, color, bg in cajas:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        facecolor=bg,
        edgecolor=color,
        linewidth=1.8,
        transform=ax2.transAxes,
    )
    ax2.add_patch(rect)
    ax2.text(
        x + w / 2,
        y + h / 2,
        txt,
        ha="center",
        va="center",
        fontsize=8,
        color=color,
        fontweight="bold",
        transform=ax2.transAxes,
        multialignment="center",
    )

# Flechas
flechas = [
    ((0.37, 0.77), (0.37, 0.57), VERDE),
    ((0.37, 0.77), (0.37, 0.42), NARANJA),
    ((0.37, 0.77), (0.37, 0.27), MORADO),
    ((0.43, 0.57), (0.55, 0.57), ROJO),
    ((0.43, 0.42), (0.55, 0.57), ROJO),
    ((0.55, 0.57), (0.55, 0.42), VERDE),
    ((0.55, 0.42), (0.55, 0.27), AZUL),
    ((0.43, 0.27), (0.55, 0.27), AZUL),
]
for (x1, y1), (x2, y2), col in flechas:
    ax2.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )

ax2.text(
    0.5,
    0.10,
    "Complejidad: O(seq^2 * d_model)",
    ha="center",
    fontsize=8.5,
    color=GRIS,
    style="italic",
    transform=ax2.transAxes,
)

plt.tight_layout()
savefig("03-atencion-qkv.png")


# ─── 04. Multi-Head Attention ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6), facecolor=FONDO)
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title(
    "Multi-Head Attention: Múltiples Perspectivas de Atención en Paralelo",
    fontsize=14,
    fontweight="bold",
    color=DARK,
    pad=10,
)

n_heads = 4
colores_heads = [AZUL, VERDE, NARANJA, MORADO]
head_labels = [
    "Head 1\n(sintaxis)",
    "Head 2\n(semántica)",
    "Head 3\n(correferencia)",
    "Head 4\n(posición)",
]

# Entrada
rect = FancyBboxPatch(
    (0.01, 0.42),
    0.10,
    0.16,
    boxstyle="round,pad=0.02",
    facecolor="#EFF6FF",
    edgecolor=AZUL,
    linewidth=2,
)
ax.add_patch(rect)
ax.text(
    0.06,
    0.50,
    "Entrada X\n[seq, d]",
    ha="center",
    va="center",
    fontsize=9,
    color=AZUL,
    fontweight="bold",
)

# Heads
for i, (col, label) in enumerate(zip(colores_heads, head_labels)):
    y_center = 0.78 - i * 0.22
    # Caja Q, K, V
    for j, (letra, offset) in enumerate([("Q", 0), ("K", 0.085), ("V", 0.17)]):
        rect = FancyBboxPatch(
            (0.17 + offset, y_center - 0.05),
            0.07,
            0.10,
            boxstyle="round,pad=0.01",
            facecolor=col + "22",
            edgecolor=col,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            0.205 + offset,
            y_center,
            letra,
            ha="center",
            va="center",
            fontsize=10,
            color=col,
            fontweight="bold",
        )

    # Atención head
    rect2 = FancyBboxPatch(
        (0.42, y_center - 0.05),
        0.12,
        0.10,
        boxstyle="round,pad=0.01",
        facecolor=col + "33",
        edgecolor=col,
        linewidth=1.8,
    )
    ax.add_patch(rect2)
    ax.text(
        0.48,
        y_center,
        "Attn\nhead",
        ha="center",
        va="center",
        fontsize=8,
        color=col,
        fontweight="bold",
    )
    ax.text(
        0.57,
        y_center,
        label,
        ha="left",
        va="center",
        fontsize=8,
        color=col,
        style="italic",
    )

    # Flechas entrada → QKV
    ax.annotate(
        "",
        xy=(0.17, y_center),
        xytext=(0.11, 0.50),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.6),
    )
    # Flecha QKV → atención
    ax.annotate(
        "",
        xy=(0.42, y_center),
        xytext=(0.345, y_center),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
    )
    # Flecha atención → concat
    ax.annotate(
        "",
        xy=(0.80, 0.50),
        xytext=(0.58, y_center),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.6),
    )

# Concat + proyeccion
rect3 = FancyBboxPatch(
    (0.80, 0.38),
    0.08,
    0.24,
    boxstyle="round,pad=0.02",
    facecolor="#FDF4FF",
    edgecolor=MORADO,
    linewidth=2.2,
)
ax.add_patch(rect3)
ax.text(
    0.84,
    0.50,
    "Concat\n+\nW_O",
    ha="center",
    va="center",
    fontsize=9,
    color=MORADO,
    fontweight="bold",
)

# Salida
rect4 = FancyBboxPatch(
    (0.91, 0.42),
    0.08,
    0.16,
    boxstyle="round,pad=0.02",
    facecolor="#F0FDF4",
    edgecolor=VERDE,
    linewidth=2,
)
ax.add_patch(rect4)
ax.text(
    0.95,
    0.50,
    "Salida\n[seq, d]",
    ha="center",
    va="center",
    fontsize=9,
    color=VERDE,
    fontweight="bold",
)
ax.annotate(
    "",
    xy=(0.91, 0.50),
    xytext=(0.88, 0.50),
    arrowprops=dict(arrowstyle="->", color=VERDE, lw=2),
)

# Nota
ax.text(
    0.5,
    0.04,
    "Cada head aprende diferentes relaciones; se concatenan y proyectan al final.\n"
    "h heads de dimension d_k = d_model / h → costo similar a 1 head de d_model",
    ha="center",
    va="center",
    fontsize=9,
    color=GRIS,
    style="italic",
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
savefig("04-multi-head-attention.png")


# ─── 05. Arquitectura Transformer Encoder ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 10), facecolor=FONDO)
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title(
    "Bloque Encoder del Transformer (BERT usa solo esto)",
    fontsize=13,
    fontweight="bold",
    color=DARK,
    pad=10,
)

bloques = [
    # (x, y, w, h, texto, color_borde, color_fondo)
    (
        0.20,
        0.88,
        0.60,
        0.08,
        "Output\n(representaciones contextuales)",
        VERDE,
        "#F0FDF4",
    ),
    (0.20, 0.76, 0.60, 0.08, "Add & Norm (Layer Normalization)", GRIS, "#F8FAFC"),
    (
        0.20,
        0.64,
        0.60,
        0.08,
        "Feed Forward Network\n(2 capas MLP con ReLU)",
        NARANJA,
        "#FFF7ED",
    ),
    (0.20, 0.52, 0.60, 0.08, "Add & Norm (Layer Normalization)", GRIS, "#F8FAFC"),
    (0.20, 0.40, 0.60, 0.08, "Multi-Head Self-Attention", AZUL, "#EFF6FF"),
    (
        0.20,
        0.28,
        0.60,
        0.08,
        "Positional Encoding\n(sin/cos o aprendido)",
        MORADO,
        "#FDF4FF",
    ),
    (0.20, 0.16, 0.60, 0.08, "Token Embeddings\n(lookup table W_E)", CIAN, "#ECFEFF"),
    (0.20, 0.04, 0.60, 0.08, "Input tokens\n(secuencia de IDs)", DARK, "#F1F5F9"),
]

for x, y, w, h, txt, color, bg in bloques:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015",
        facecolor=bg,
        edgecolor=color,
        linewidth=2,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        txt,
        ha="center",
        va="center",
        fontsize=9,
        color=color,
        fontweight="bold",
        transform=ax.transAxes,
        multialignment="center",
    )

# Flechas verticales
for y_from, y_to in [
    (0.12, 0.16),
    (0.24, 0.28),
    (0.36, 0.40),
    (0.48, 0.52),
    (0.60, 0.64),
    (0.72, 0.76),
    (0.84, 0.88),
]:
    ax.annotate(
        "",
        xy=(0.5, y_to),
        xytext=(0.5, y_from),
        arrowprops=dict(arrowstyle="->", color=GRIS, lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )

# Skip connections
for (x1, y1), (x2, y2), label in [
    ((0.82, 0.40), (0.82, 0.52), "residual"),
    ((0.82, 0.64), (0.82, 0.76), "residual"),
]:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.8, alpha=0.7),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    ax.text(
        0.845,
        (y1 + y2) / 2,
        label,
        ha="left",
        va="center",
        fontsize=7.5,
        color=ROJO,
        style="italic",
        transform=ax.transAxes,
    )

ax.text(
    0.02,
    0.5,
    "x N bloques",
    ha="left",
    va="center",
    fontsize=10,
    color=GRIS,
    style="italic",
    rotation=90,
    transform=ax.transAxes,
)

savefig("05-arquitectura-transformer.png")


# ─── 06. BERT vs GPT: entrenamiento y uso ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor=FONDO)
fig.suptitle(
    "BERT vs GPT: Diferencias Fundamentales de Diseño",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

# Panel izquierdo: comparación visual
ax = axes[0]
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title("Arquitectura y preentrenamiento", fontweight="bold", color=DARK, pad=8)

items_bert = [
    ("Solo Encoder (bidireccional)", AZUL),
    ("Preentrenamiento: MLM + NSP", AZUL),
    ("MLM: predice tokens [MASK]", AZUL),
    ("Ve contexto izq. Y derecho", AZUL),
    ("Ideal para: clasificacion,\n NER, QA extractivo", VERDE),
    ("Modelos: BERT, RoBERTa,\n ALBERT, DeBERTa", AZUL),
]
items_gpt = [
    ("Solo Decoder (causal/autoregresivo)", MORADO),
    ("Preentrenamiento: LM causal", MORADO),
    ("LM: predice el siguiente token", MORADO),
    ("Solo ve contexto izquierdo", MORADO),
    ("Ideal para: generacion, chat,\n completado de texto", NARANJA),
    ("Modelos: GPT-2/3/4, LLaMA,\n Mistral, Qwen", MORADO),
]

y_start = 0.90
ax.text(
    0.12,
    y_start + 0.04,
    "BERT / Encoder",
    ha="center",
    fontsize=12,
    fontweight="bold",
    color=AZUL,
    transform=ax.transAxes,
)
ax.text(
    0.62,
    y_start + 0.04,
    "GPT / Decoder",
    ha="center",
    fontsize=12,
    fontweight="bold",
    color=MORADO,
    transform=ax.transAxes,
)

for i, ((txt, col_b), (txt_g, col_g)) in enumerate(zip(items_bert, items_gpt)):
    y = y_start - i * 0.16
    # BERT
    rect_b = FancyBboxPatch(
        (0.01, y - 0.08),
        0.44,
        0.10,
        boxstyle="round,pad=0.01",
        facecolor=col_b + "22",
        edgecolor=col_b,
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(rect_b)
    ax.text(
        0.23,
        y - 0.025,
        txt,
        ha="center",
        va="center",
        fontsize=8,
        color=col_b,
        transform=ax.transAxes,
        multialignment="center",
    )
    # GPT
    rect_g = FancyBboxPatch(
        (0.50, y - 0.08),
        0.44,
        0.10,
        boxstyle="round,pad=0.01",
        facecolor=col_g + "22",
        edgecolor=col_g,
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(rect_g)
    ax.text(
        0.72,
        y - 0.025,
        txt_g,
        ha="center",
        va="center",
        fontsize=8,
        color=col_g,
        transform=ax.transAxes,
        multialignment="center",
    )

ax.text(
    0.50,
    0.03,
    "VS",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color=GRIS,
    transform=ax.transAxes,
)

# Panel derecho: curvas fine-tuning BERT
ax2 = axes[1]
ax2.set_facecolor(FONDO)
epochs = np.arange(1, 11)
train_loss = 0.65 * np.exp(-epochs * 0.25) + 0.08
val_loss = 0.70 * np.exp(-epochs * 0.20) + 0.12
train_acc = 1 - 0.62 * np.exp(-epochs * 0.28)
val_acc = 1 - 0.60 * np.exp(-epochs * 0.22)

ax2_twin = ax2.twinx()
ax2.plot(
    epochs, train_loss, "o-", color=ROJO, lw=2.5, ms=5, label="Train Loss", alpha=0.9
)
ax2.plot(
    epochs, val_loss, "s--", color=NARANJA, lw=2.5, ms=5, label="Val Loss", alpha=0.9
)
ax2_twin.plot(epochs, train_acc, "^-", color=AZUL, lw=2.5, ms=5, label="Train Acc")
ax2_twin.plot(epochs, val_acc, "D--", color=VERDE, lw=2.5, ms=5, label="Val Acc")

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss", color=ROJO)
ax2_twin.set_ylabel("Accuracy", color=AZUL)
ax2.set_title(
    "Fine-tuning BERT para clasificacion\n(LR=2e-5, 3-5 epochs recomendados)",
    fontweight="bold",
    color=DARK,
)

# Zona recomendada (epochs 2-4)
ax2.axvspan(2, 4, alpha=0.08, color=VERDE, label="Zona optima")
ax2.text(
    3.0, 0.50, "Zona\noptima", ha="center", fontsize=8.5, color=VERDE, fontweight="bold"
)
ax2.set_xlim(0.5, 10.5)
ax2.set_ylim(0, 0.8)
ax2_twin.set_ylim(0.4, 1.05)
ax2.grid(True, alpha=0.3)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

plt.tight_layout()
savefig("06-bert-vs-gpt.png")


# ─── 07. Tokenización subword: WordPiece / BPE ────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor=FONDO)
fig.suptitle(
    "Tokenización Subword: Por qué los Transformers no tienen OOV",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

# Panel superior: tabla comparativa de tokenización
ax = axes[0]
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title(
    "Comparación por tipo de tokenización", fontweight="bold", color=DARK, pad=6
)

palabras_tok = [
    "clasificación",
    "descontextualizado",
    "NLPBolivia2024",
    "supermaravilloso",
    "unknownword123",
]
tok_word = [
    ["clasificación"],
    ["descontextualizado"],
    ["UNK"],
    ["UNK"],
    ["UNK"],
]
tok_bpe = [
    ["clas", "if", "ica", "ción"],
    ["des", "con", "text", "ual", "izado"],
    ["NLP", "Bo", "livia", "2024"],
    ["super", "mara", "vill", "oso"],
    ["un", "known", "word", "123"],
]
tok_char = [
    list("clasif..."),  # truncado
    list("descon..."),
    list("NLPBol..."),
    list("superm..."),
    list("unknow..."),
]

cols = [
    "Palabra original",
    "Word tokenizer (OOV=UNK)",
    "BPE / WordPiece",
    "Char (parcial)",
]
col_x = [0.01, 0.20, 0.50, 0.82]
col_colors = [DARK, ROJO, VERDE, NARANJA]

for cx, col, cc in zip(col_x, cols, col_colors):
    ax.text(
        cx,
        0.96,
        col,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=cc,
        transform=ax.transAxes,
    )

for i, (pal, tw, tb, tc) in enumerate(zip(palabras_tok, tok_word, tok_bpe, tok_char)):
    y = 0.82 - i * 0.18
    bg = "#F1F5F9" if i % 2 == 0 else FONDO
    rect = FancyBboxPatch(
        (0, y - 0.07),
        1.0,
        0.14,
        boxstyle="round,pad=0.01",
        facecolor=bg,
        edgecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(
        0.01,
        y,
        pal,
        ha="left",
        va="center",
        fontsize=9,
        color=DARK,
        family="monospace",
        transform=ax.transAxes,
    )
    # Word
    tw_str = " | ".join(tw)
    col_tw = ROJO if "UNK" in tw_str else DARK
    ax.text(
        0.20,
        y,
        tw_str,
        ha="left",
        va="center",
        fontsize=8.5,
        color=col_tw,
        family="monospace",
        transform=ax.transAxes,
    )
    # BPE
    bpe_str = " | ".join(f"[{t}]" for t in tb)
    ax.text(
        0.50,
        y,
        bpe_str,
        ha="left",
        va="center",
        fontsize=8,
        color=VERDE,
        family="monospace",
        transform=ax.transAxes,
    )
    # Char
    tc_str = " ".join(tc[:6]) + " ..."
    ax.text(
        0.82,
        y,
        tc_str,
        ha="left",
        va="center",
        fontsize=7.5,
        color=NARANJA,
        family="monospace",
        transform=ax.transAxes,
    )

# Panel inferior: longitud de secuencia por tipo de tokenización
ax2 = axes[1]
ax2.set_facecolor(FONDO)

textos_len = [
    "Hola",
    "El gato come",
    "Los modelos de machine learning",
    "Implementando transformers en produccion con PyTorch",
    "Entrenamiento de modelos de lenguaje grande con tecnicas de fine-tuning eficiente",
]
lens_word = [len(t.split()) for t in textos_len]
lens_bpe = [int(l * 1.35) for l in lens_word]  # BPE ~35% mas tokens
lens_char = [len(t.replace(" ", "")) for t in textos_len]

x = np.arange(len(textos_len))
w = 0.25
ax2.bar(
    x - w, lens_word, w, label="Word tokens", color=ROJO, alpha=0.85, edgecolor="white"
)
ax2.bar(x, lens_bpe, w, label="BPE tokens", color=VERDE, alpha=0.85, edgecolor="white")
ax2.bar(
    x + w,
    lens_char,
    w,
    label="Char tokens",
    color=NARANJA,
    alpha=0.85,
    edgecolor="white",
)
ax2.set_xticks(x)
ax2.set_xticklabels(
    [t[:30] + "..." if len(t) > 30 else t for t in textos_len],
    rotation=20,
    ha="right",
    fontsize=8,
)
ax2.set_ylabel("Longitud de secuencia")
ax2.set_title(
    "Longitud de secuencia según tipo de tokenización",
    fontweight="bold",
    color=DARK,
    pad=6,
)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
savefig("07-tokenizacion-subword.png")


# ─── 08. Fine-tuning BERT: impacto de hiperparámetros ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=FONDO)
fig.suptitle(
    "Fine-tuning BERT: Impacto de Hiperparámetros Clave",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

# Subgráfico 1: LR vs F1
ax1 = axes[0]
ax1.set_facecolor(FONDO)
lrs = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 5e-4]
f1s = [0.71, 0.77, 0.82, 0.86, 0.89, 0.86, 0.78, 0.65]
ax1.semilogx(lrs, f1s, "o-", color=AZUL, lw=2.5, ms=7)
best_lr = lrs[f1s.index(max(f1s))]
ax1.axvline(x=best_lr, color=VERDE, lw=2, ls="--", alpha=0.8)
ax1.text(best_lr * 1.5, 0.70, f"Optimo\n{best_lr:.0e}", fontsize=8.5, color=VERDE)
ax1.fill_between(
    [1e-5, 5e-5],
    [0.62] * 2,
    [0.92] * 2,
    alpha=0.1,
    color=VERDE,
    label="Rango recomendado",
)
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("F1-macro")
ax1.set_title("LR vs F1", fontweight="bold", color=DARK)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.60, 0.95)

# Subgráfico 2: epochs vs F1 para distintos tamaños de dataset
ax2 = axes[1]
ax2.set_facecolor(FONDO)
epochs_arr = np.arange(1, 11)
configs = [
    (
        "500 ejemplos",
        [0.65, 0.70, 0.73, 0.72, 0.70, 0.68, 0.65, 0.63, 0.60, 0.58],
        ROJO,
    ),
    (
        "2k ejemplos",
        [0.72, 0.79, 0.83, 0.85, 0.85, 0.84, 0.82, 0.80, 0.78, 0.76],
        NARANJA,
    ),
    (
        "10k ejemplos",
        [0.78, 0.84, 0.87, 0.89, 0.90, 0.90, 0.89, 0.88, 0.87, 0.86],
        VERDE,
    ),
    (
        "50k ejemplos",
        [0.82, 0.87, 0.90, 0.91, 0.92, 0.92, 0.92, 0.91, 0.91, 0.90],
        AZUL,
    ),
]
for label, vals, col in configs:
    ax2.plot(epochs_arr, vals, "o-", color=col, lw=2, ms=5, label=label)
ax2.axvline(x=3, color=GRIS, lw=1.5, ls="--", alpha=0.7, label="3 epochs (heuristica)")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("F1-macro (val)")
ax2.set_title("Epochs según tamaño de dataset", fontweight="bold", color=DARK)
ax2.legend(fontsize=7.5)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.55, 0.95)

# Subgráfico 3: batch size vs tiempo/GPU vs F1
ax3 = axes[2]
ax3.set_facecolor(FONDO)
bs_vals = [8, 16, 32, 64, 128]
f1_bs = [0.88, 0.89, 0.89, 0.87, 0.84]
time_bs = [120, 65, 38, 28, 22]  # segundos/epoch (relativo)

ax3_twin = ax3.twinx()
ax3.bar(range(len(bs_vals)), f1_bs, color=AZUL, alpha=0.7, label="F1-macro", width=0.4)
ax3_twin.plot(
    range(len(bs_vals)),
    time_bs,
    "s-",
    color=ROJO,
    lw=2.5,
    ms=7,
    label="Tiempo (seg/epoch)",
)
ax3.set_xticks(range(len(bs_vals)))
ax3.set_xticklabels([f"bs={b}" for b in bs_vals], fontsize=9)
ax3.set_ylabel("F1-macro", color=AZUL)
ax3_twin.set_ylabel("Tiempo (seg/epoch)", color=ROJO)
ax3.set_title("Batch size: F1 vs tiempo", fontweight="bold", color=DARK)
ax3.set_ylim(0.75, 0.95)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
savefig("08-finetuning-hiperparametros.png")


# ─── 09. Dashboard resumen ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=FONDO)
fig.suptitle(
    "Tema 15: Embeddings y Transformers — Dashboard Resumen",
    fontsize=16,
    fontweight="bold",
    color=DARK,
    y=0.97,
)

gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.45,
    wspace=0.42,
    left=0.06,
    right=0.97,
    top=0.91,
    bottom=0.06,
)

# Panel 1: Evolucion de representaciones NLP
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title(
    "Evolución de Representaciones", fontsize=10, fontweight="bold", color=DARK
)
representaciones = [
    ("One-Hot", "1995", "Binario, sparse,\nno similaridad", GRIS),
    ("Word2Vec", "2013", "Denso, analogias,\nno contexto", AZUL),
    ("GloVe", "2014", "Co-ocurrencia global,\nno contexto", CIAN),
    ("ELMo", "2018", "Contextual,\nbidireccional LSTM", VERDE),
    ("BERT", "2018", "Transformer,\nMasked LM, bidirec.", MORADO),
    ("GPT-4/LLaMA", "2023", "Autoregresivo,\nmultimodal, RLHF", NARANJA),
]
y = 0.95
for nombre, año, desc, col in representaciones:
    ax1.text(0.0, y, f"{año}", fontsize=7.5, color=GRIS, transform=ax1.transAxes)
    ax1.text(
        0.14,
        y,
        nombre,
        fontsize=9,
        fontweight="bold",
        color=col,
        transform=ax1.transAxes,
    )
    ax1.text(0.14, y - 0.07, desc, fontsize=7, color=DARK, transform=ax1.transAxes)
    y -= 0.18

# Panel 2: Benchmarks de modelos
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(FONDO)
modelos_bench = [
    "TF-IDF\n+LogReg",
    "Word2Vec\n+MLP",
    "BERT\nbase",
    "RoBERTa\nlarge",
    "DeBERTa\nv3",
]
acc_bench = [0.85, 0.88, 0.92, 0.94, 0.96]
params_m = [0.0, 20, 110, 355, 183]  # millones de parametros

colors_bench = [GRIS, CIAN, AZUL, VERDE, MORADO]
bars = ax2.bar(
    modelos_bench, acc_bench, color=colors_bench, alpha=0.85, edgecolor="white"
)
ax2.set_ylabel("F1 / Accuracy")
ax2.set_ylim(0.75, 1.0)
ax2.set_title(
    "Rendimiento en clasificacion\nde texto (benchmark)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax2.grid(axis="y", alpha=0.3)
for bar, acc, p in zip(bars, acc_bench, params_m):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.002,
        f"{acc:.0%}\n({p}M)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )

# Panel 3: Receta de fine-tuning
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(FONDO)
ax3.axis("off")
ax3.set_title("Receta de Fine-Tuning BERT", fontsize=10, fontweight="bold", color=DARK)
pasos_ft = [
    ("1.", "Elige modelo base\n(bert-base-multilingual si es ES)", AZUL),
    ("2.", "LR = 2e-5 a 5e-5\n(usa scheduler warmup)", VERDE),
    ("3.", "Batch size = 16 o 32\n(acumular gradientes si poco RAM)", MORADO),
    ("4.", "Max epochs = 3-5\n(early stopping en val loss)", NARANJA),
    ("5.", "Warmup = 6% del total\nde pasos de entrenamiento", CIAN),
    ("6.", "Weight decay = 0.01\n(AdamW, no Adam)", ROJO),
]
y = 0.93
for num, texto, col in pasos_ft:
    ax3.text(
        0.0, y, num, fontsize=10, fontweight="bold", color=col, transform=ax3.transAxes
    )
    ax3.text(0.10, y, texto, fontsize=8, color=DARK, transform=ax3.transAxes)
    y -= 0.16

# Panel 4: Embedding 2D con PCA (simulado)
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor(FONDO)
np.random.seed(7)
grupos = {
    "Tecnologia": (["IA", "modelo", "red", "datos", "GPU", "Python"], AZUL, (0.6, 0.6)),
    "Animales": (["gato", "perro", "pez", "ave", "leon", "tigre"], VERDE, (-0.6, 0.4)),
    "Comida": (
        ["pizza", "arroz", "sopa", "pan", "leche", "cafe"],
        NARANJA,
        (-0.5, -0.6),
    ),
    "Politica": (["voto", "ley", "estado", "gobierno", "partido"], MORADO, (0.4, -0.5)),
}
for grupo, (palabras, col, centro) in grupos.items():
    xs = np.random.normal(centro[0], 0.20, len(palabras))
    ys = np.random.normal(centro[1], 0.15, len(palabras))
    ax4.scatter(xs, ys, color=col, s=80, alpha=0.8, zorder=5)
    for w, x, y in zip(palabras, xs, ys):
        ax4.text(x + 0.02, y + 0.03, w, fontsize=8, color=col, fontweight="bold")
    # Elipse de grupo
    from matplotlib.patches import Ellipse

    ellipse = Ellipse(
        xy=centro,
        width=0.7,
        height=0.5,
        angle=0,
        edgecolor=col,
        facecolor=col,
        alpha=0.07,
        lw=1.5,
    )
    ax4.add_patch(ellipse)
    ax4.text(
        centro[0],
        centro[1] + 0.32,
        grupo,
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=col,
        alpha=0.8,
    )

ax4.set_xlabel("Componente Principal 1 (PCA)")
ax4.set_ylabel("Componente Principal 2 (PCA)")
ax4.set_title(
    "Proyección 2D de Embeddings: Clusters Semánticos",
    fontweight="bold",
    fontsize=10,
    color=DARK,
)
ax4.axhline(0, color=GRIS, lw=0.8, alpha=0.4)
ax4.axvline(0, color=GRIS, lw=0.8, alpha=0.4)
ax4.set_xlim(-1.1, 1.1)
ax4.set_ylim(-1.0, 1.0)
ax4.grid(True, alpha=0.15)

# Panel 5: Modelos multilingues
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(FONDO)
modelos_mult = ["mBERT", "XLM-R\nbase", "XLM-R\nlarge", "mT5\nsmall", "LaBSE"]
langs = [104, 100, 100, 101, 109]  # idiomas soportados
f1_mult = [0.82, 0.86, 0.90, 0.84, 0.85]

scatter = ax5.scatter(
    langs,
    f1_mult,
    c=[AZUL, VERDE, MORADO, NARANJA, CIAN],
    s=200,
    zorder=5,
    edgecolors="white",
    linewidth=1.5,
)
for m, l, f in zip(modelos_mult, langs, f1_mult):
    ax5.text(l + 0.5, f + 0.002, m, fontsize=8, color=DARK)
ax5.set_xlabel("Idiomas soportados")
ax5.set_ylabel("F1 promedio (multilingue)")
ax5.set_title(
    "Modelos multilingues:\nidiomas vs rendimiento",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(0.78, 0.94)

savefig("09-dashboard.png")

print("\nTodos los graficos del tema 15 generados en:", OUT)
