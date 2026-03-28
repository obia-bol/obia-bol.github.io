"""
gen_tema14.py — Genera 9 gráficos para el tema 14: Fundamentos de NLP
Salida: public/ruta-aprendizaje-graficos/tema-14/
"""

import os, re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

OUT = "public/ruta-aprendizaje-graficos/tema-14"
os.makedirs(OUT, exist_ok=True)

AZUL = "#2563EB"
VERDE = "#16A34A"
ROJO = "#DC2626"
NARANJA = "#EA580C"
MORADO = "#7C3AED"
GRIS = "#6B7280"
AMARILLO = "#D97706"
CIAN = "#0891B2"
FONDO = "#F8FAFC"
DARK = "#1E293B"


def savefig(name):
    plt.savefig(f"{OUT}/{name}", dpi=130, bbox_inches="tight", facecolor=FONDO)
    plt.close()
    print(f"  ✓ {name}")


# ─── 01. Pipeline de preprocesamiento de texto ────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5), facecolor=FONDO)
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title(
    "Pipeline de Preprocesamiento de Texto",
    fontsize=15,
    fontweight="bold",
    color=DARK,
    pad=12,
)

pasos = [
    ("Texto\noriginal", "#EFF6FF", AZUL, '"¡Los    Árboles SON\nbonitos!"'),
    ("Lowercase\n+ unicode", "#F0FDF4", VERDE, '"los arboles son\nbonitos"'),
    ("Eliminar\npuntuacion", "#FFF7ED", NARANJA, '"los arboles son\nbonitos"'),
    ("Tokenizar", "#FDF4FF", MORADO, "['los', 'arboles',\n'son', 'bonitos']"),
    ("Stopwords", "#FEF2F2", ROJO, "['arboles',\n'bonitos']"),
    ("Stemming /\nLematizar", "#ECFDF5", CIAN, "['arbol',\n'bonito']"),
]

n = len(pasos)
W, H, gap = 1.7, 1.1, 0.25
total = n * W + (n - 1) * gap
x0 = (13 - total) / 2

for i, (titulo, bg, color, ejemplo) in enumerate(pasos):
    x = x0 + i * (W + gap)
    rect = FancyBboxPatch(
        (x, 1.7),
        W,
        H,
        boxstyle="round,pad=0.06",
        facecolor=bg,
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + W / 2,
        1.7 + H / 2 + 0.1,
        titulo,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=color,
        multialignment="center",
    )
    ax.text(
        x + W / 2,
        1.55,
        ejemplo,
        ha="center",
        va="top",
        fontsize=7,
        color=GRIS,
        multialignment="center",
        family="monospace",
    )
    # flecha
    if i < n - 1:
        ax.annotate(
            "",
            xy=(x + W + gap, 1.7 + H / 2),
            xytext=(x + W, 1.7 + H / 2),
            arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.5),
        )

ax.set_xlim(0, 13)
ax.set_ylim(0.5, 3.2)
savefig("01-pipeline-preprocesamiento.png")


# ─── 02. Bag of Words vs TF-IDF ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=FONDO)
fig.suptitle(
    "Bag of Words vs TF-IDF: Representación de Documentos",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

corpus = [
    "el gato come pescado",
    "el perro come carne",
    "el gato y el perro juegan",
]
vocab = sorted(set(" ".join(corpus).split()))
# BoW
bow = np.zeros((3, len(vocab)), dtype=int)
for i, doc in enumerate(corpus):
    for w in doc.split():
        bow[i, vocab.index(w)] += 1

ax = axes[0]
ax.set_facecolor(FONDO)
im = ax.imshow(bow, cmap="Blues", aspect="auto", vmin=0, vmax=2)
ax.set_xticks(range(len(vocab)))
ax.set_xticklabels(vocab, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels([f"Doc {i+1}" for i in range(3)], fontsize=9)
for i in range(3):
    for j in range(len(vocab)):
        ax.text(
            j,
            i,
            str(bow[i, j]),
            ha="center",
            va="center",
            fontsize=10,
            color="white" if bow[i, j] > 1 else DARK,
        )
ax.set_title("Bag of Words (conteo)", fontweight="bold", color=AZUL, pad=8)
plt.colorbar(im, ax=ax, shrink=0.8)

# TF-IDF simplificado
tf = bow / (bow.sum(axis=1, keepdims=True) + 1e-9)
n_docs = 3
df = (bow > 0).sum(axis=0)
idf = np.log(n_docs / (df + 1)) + 1
tfidf = tf * idf

ax2 = axes[1]
ax2.set_facecolor(FONDO)
im2 = ax2.imshow(tfidf, cmap="Greens", aspect="auto", vmin=0)
ax2.set_xticks(range(len(vocab)))
ax2.set_xticklabels(vocab, rotation=45, ha="right", fontsize=9)
ax2.set_yticks(range(3))
ax2.set_yticklabels([f"Doc {i+1}" for i in range(3)], fontsize=9)
for i in range(3):
    for j in range(len(vocab)):
        v = tfidf[i, j]
        ax2.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if v > 0.15 else DARK,
        )
ax2.set_title("TF-IDF (peso por rareza)", fontweight="bold", color=VERDE, pad=8)
plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
savefig("02-bow-tfidf.png")


# ─── 03. Stopwords: antes y después ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=FONDO)
fig.suptitle(
    "Efecto de Eliminar Stopwords en Frecuencia de Términos",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

texto = (
    "el modelo de aprendizaje automatico es un sistema que aprende de los "
    "datos para hacer predicciones sobre nuevos datos de entrada en el problema"
)
tokens = texto.split()
stopwords_es = {
    "el",
    "de",
    "es",
    "un",
    "que",
    "los",
    "para",
    "sobre",
    "en",
    "a",
    "y",
    "se",
    "la",
    "del",
    "con",
    "una",
}

from collections import Counter

freq_all = Counter(tokens).most_common(12)
freq_clean = Counter(t for t in tokens if t not in stopwords_es).most_common(12)

for ax, (freq, titulo, color) in zip(
    axes,
    [
        (freq_all, "Con stopwords", ROJO),
        (freq_clean, "Sin stopwords", VERDE),
    ],
):
    ax.set_facecolor(FONDO)
    words, counts = zip(*freq)
    colors = [ROJO if w in stopwords_es else AZUL for w in words]
    bars = ax.barh(
        range(len(words)),
        counts,
        color=colors if color == ROJO else VERDE,
        edgecolor="white",
    )
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Frecuencia")
    ax.set_title(titulo, fontweight="bold", color=color, pad=8)
    ax.set_facecolor(FONDO)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )

axes[0].legend(
    handles=[
        mpatches.Patch(color=ROJO, label="Stopword"),
        mpatches.Patch(color=AZUL, label="Término útil"),
    ],
    loc="lower right",
    fontsize=8,
)
plt.tight_layout()
savefig("03-stopwords.png")


# ─── 04. Stemming vs Lematización ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6), facecolor=FONDO)
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title(
    "Stemming vs Lematización: Reducción de Variantes de Palabras",
    fontsize=14,
    fontweight="bold",
    color=DARK,
    pad=12,
)

ejemplos = [
    ("corriendo", "corr", "correr"),
    ("corremos", "corr", "correr"),
    ("corriste", "corr", "correr"),
    ("mejores", "mejor", "bueno"),
    ("mejorando", "mejor", "mejorar"),
    ("estudiando", "estudi", "estudiar"),
    ("estudiante", "estudi", "estudiante"),
    ("estudios", "estudi", "estudio"),
]

cols = ["Forma original", "Stemming (Porter)", "Lematización (spaCy)"]
col_x = [1, 4.5, 8]
col_colors = [DARK, NARANJA, VERDE]

for cx, col, cc in zip(col_x, cols, col_colors):
    ax.text(
        cx, 5.5, col, ha="left", va="center", fontsize=11, fontweight="bold", color=cc
    )

for i, (orig, stem, lema) in enumerate(ejemplos):
    y = 4.8 - i * 0.55
    bg = "#F1F5F9" if i % 2 == 0 else FONDO
    rect = FancyBboxPatch(
        (0.5, y - 0.2),
        10,
        0.45,
        boxstyle="round,pad=0.04",
        facecolor=bg,
        edgecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        1,
        y + 0.05,
        orig,
        ha="left",
        va="center",
        fontsize=10,
        color=DARK,
        family="monospace",
    )
    ax.text(
        4.5,
        y + 0.05,
        stem,
        ha="left",
        va="center",
        fontsize=10,
        color=NARANJA,
        family="monospace",
    )
    ax.text(
        8,
        y + 0.05,
        lema,
        ha="left",
        va="center",
        fontsize=10,
        color=VERDE,
        family="monospace",
    )

# Anotaciones
ax.text(
    4.5,
    0.2,
    "❌ Puede crear formas\nno existentes",
    ha="left",
    va="center",
    fontsize=9,
    color=ROJO,
    style="italic",
)
ax.text(
    8,
    0.2,
    "✓ Siempre produce\nformas válidas",
    ha="left",
    va="center",
    fontsize=9,
    color=VERDE,
    style="italic",
)

ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
savefig("04-stemming-lematizacion.png")


# ─── 05. N-gramas: distribución de frecuencias ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=FONDO)
fig.suptitle(
    "N-gramas: Capturando Contexto Local en Texto",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

texto2 = (
    "el modelo aprende de los datos el modelo mejora con mas datos "
    "el sistema de aprendizaje automatico necesita muchos datos buenos"
)
tokens2 = texto2.split()


def get_ngrams(tokens, n):
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


for ax, (n, titulo, color) in zip(
    axes,
    [
        (1, "Unigramas (1-gram)", AZUL),
        (2, "Bigramas (2-gram)", VERDE),
        (3, "Trigramas (3-gram)", MORADO),
    ],
):
    ax.set_facecolor(FONDO)
    ngrams = get_ngrams(tokens2, n)
    freq = Counter(ngrams).most_common(8)
    if freq:
        words, counts = zip(*freq)
        bars = ax.barh(
            range(len(words)), counts, color=color, edgecolor="white", alpha=0.85
        )
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8 if n > 1 else 9)
        ax.invert_yaxis()
        ax.set_title(titulo, fontweight="bold", color=color, pad=6)
        ax.set_xlabel("Frecuencia")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=8,
            )

plt.tight_layout()
savefig("05-ngramas.png")


# ─── 06. TF-IDF Heatmap en corpus real ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6), facecolor=FONDO)
ax.set_facecolor(FONDO)

docs = {
    "Reseña +1": "excelente producto muy bueno recomendado calidad perfecta",
    "Reseña +2": "increible calidad muy recomendado producto excelente servicio",
    "Reseña -1": "terrible producto malo defectuoso no funciona pesimo",
    "Reseña -2": "muy malo pesimo no recomendado defectuoso producto roto",
    "Neutral 1": "producto recibido entrega rapida embalaje correcto",
    "Neutral 2": "producto llegó bien embalaje normal entrega correcta",
}

all_tokens = []
for d in docs.values():
    all_tokens.extend(d.split())
stopwords = {"muy", "no", "y", "de", "el", "la", "en", "a", "con", "del"}
vocab_tfidf = [w for w, _ in Counter(all_tokens).most_common(14) if w not in stopwords]

doc_names = list(docs.keys())
n_d = len(doc_names)
bow2 = np.zeros((n_d, len(vocab_tfidf)))
for i, text in enumerate(docs.values()):
    for w in text.split():
        if w in vocab_tfidf:
            bow2[i, vocab_tfidf.index(w)] += 1

tf2 = bow2 / (bow2.sum(axis=1, keepdims=True) + 1e-9)
df2 = (bow2 > 0).sum(axis=0)
idf2 = np.log(n_d / (df2 + 1)) + 1
tfidf2 = tf2 * idf2

im = ax.imshow(tfidf2, cmap="RdYlGn", aspect="auto", vmin=0, vmax=tfidf2.max())
ax.set_xticks(range(len(vocab_tfidf)))
ax.set_xticklabels(vocab_tfidf, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n_d))
ax.set_yticklabels(doc_names, fontsize=9)

for i in range(n_d):
    for j in range(len(vocab_tfidf)):
        v = tfidf2[i, j]
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="black" if 0.05 < v < 0.25 else "white",
        )

plt.colorbar(im, ax=ax, label="Peso TF-IDF", shrink=0.8)
ax.set_title(
    "TF-IDF Heatmap: Corpus de Reseñas (positivas/negativas/neutras)",
    fontsize=13,
    fontweight="bold",
    color=DARK,
    pad=10,
)
plt.tight_layout()
savefig("06-tfidf-heatmap.png")


# ─── 07. Pipeline de clasificación TF-IDF + Regresión Logística ───────────────
fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=FONDO)
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title(
    "Pipeline de Clasificación de Texto: TF-IDF + Regresión Logística",
    fontsize=14,
    fontweight="bold",
    color=DARK,
    pad=10,
)

etapas = [
    ("Texto\ncrudo", AZUL, "#EFF6FF", '"Gran producto\nlo recomiendo"'),
    ("Limpieza\n+ Tokens", VERDE, "#F0FDF4", "['gran','producto',\n'recomiendo']"),
    ("TF-IDF\nVectorizer", MORADO, "#FDF4FF", "[0.0, 0.42, 0.0,\n0.61, 0.0, ...]"),
    ("Logistic\nRegression", NARANJA, "#FFF7ED", "P(pos)=0.87\nP(neg)=0.13"),
    ("Predicción\nFinal", VERDE, "#F0FDF4", "⭐ Positivo\n(confianza 87%)"),
]

W, H, gap = 1.9, 1.1, 0.3
n = len(etapas)
total = n * W + (n - 1) * gap
x0 = (13 - total) / 2

for i, (titulo, color, bg, detalle) in enumerate(etapas):
    x = x0 + i * (W + gap)
    rect = FancyBboxPatch(
        (x, 2.0),
        W,
        H,
        boxstyle="round,pad=0.07",
        facecolor=bg,
        edgecolor=color,
        linewidth=2.2,
    )
    ax.add_patch(rect)
    ax.text(
        x + W / 2,
        2.0 + H / 2 + 0.1,
        titulo,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=color,
        multialignment="center",
    )
    ax.text(
        x + W / 2,
        1.88,
        detalle,
        ha="center",
        va="top",
        fontsize=7.2,
        color=GRIS,
        multialignment="center",
        family="monospace",
    )
    if i < n - 1:
        ax.annotate(
            "",
            xy=(x + W + gap, 2.0 + H / 2),
            xytext=(x + W, 2.0 + H / 2),
            arrowprops=dict(arrowstyle="->", color=GRIS, lw=2),
        )

# metricas abajo
metricas = [
    ("Pipeline de sklearn", AZUL, 3.5, 1.1),
    ("Accuracy: 85-90%", VERDE, 6.5, 1.1),
    ("F1-macro: 0.83", MORADO, 9.5, 1.1),
    ("Fit time: < 1 seg", NARANJA, 3.5, 0.55),
    ("Sparse matrix", CIAN, 6.5, 0.55),
    ("Interpretable coef", ROJO, 9.5, 0.55),
]
for texto, col, x, y in metricas:
    ax.text(
        x,
        y,
        f"✓ {texto}",
        ha="center",
        va="center",
        fontsize=8.5,
        color=col,
        fontweight="bold",
    )

ax.set_xlim(0, 13)
ax.set_ylim(0.3, 3.5)
savefig("07-pipeline-clasificacion.png")


# ─── 08. BoW+LogReg vs Transformer (curvas de aprendizaje) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=FONDO)
fig.suptitle(
    "Baseline Clásico vs Transformer: Curvas de Aprendizaje por Tamaño de Dataset",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

n_samples = np.array([100, 500, 1000, 2000, 5000, 10000, 50000])

# BoW + LogReg: excelente con pocos datos, satura pronto
np.random.seed(42)
bow_acc = np.array([0.62, 0.74, 0.79, 0.82, 0.85, 0.86, 0.87])

# Transformer: necesita mas datos pero supera con suficientes
trans_acc = np.array([0.55, 0.70, 0.78, 0.84, 0.89, 0.91, 0.93])

ax = axes[0]
ax.set_facecolor(FONDO)
ax.semilogx(n_samples, bow_acc, "o-", color=AZUL, lw=2.5, ms=6, label="TF-IDF + LogReg")
ax.semilogx(
    n_samples, trans_acc, "s-", color=MORADO, lw=2.5, ms=6, label="BERT fine-tuned"
)
ax.axvline(x=2000, color=GRIS, lw=1.5, ls="--", alpha=0.7)
ax.text(2200, 0.56, "Punto de\ncruce ~2k", fontsize=8, color=GRIS)
ax.fill_between(
    n_samples,
    bow_acc,
    trans_acc,
    where=bow_acc >= trans_acc,
    alpha=0.12,
    color=AZUL,
    label="BoW domina",
)
ax.fill_between(
    n_samples,
    bow_acc,
    trans_acc,
    where=bow_acc < trans_acc,
    alpha=0.12,
    color=MORADO,
    label="BERT domina",
)
ax.set_xlabel("Tamaño del dataset (escala log)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy por tamaño de datos", fontweight="bold", color=DARK)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.0)

# Comparacion de recursos
ax2 = axes[1]
ax2.set_facecolor(FONDO)
categorias = [
    "Accuracy\n(10k docs)",
    "Tiempo\nentrenamiento",
    "Memoria\nRAM",
    "Interpretabilidad",
    "Deploy\nsencillo",
]
bow_vals = [0.86, 0.99, 0.95, 0.95, 0.99]  # relativo (mayor = mejor en esa categoria)
bert_vals = [0.91, 0.25, 0.30, 0.40, 0.60]

x_pos = np.arange(len(categorias))
width = 0.35
bars1 = ax2.bar(
    x_pos - width / 2,
    bow_vals,
    width,
    label="TF-IDF + LogReg",
    color=AZUL,
    alpha=0.85,
    edgecolor="white",
)
bars2 = ax2.bar(
    x_pos + width / 2,
    bert_vals,
    width,
    label="BERT fine-tuned",
    color=MORADO,
    alpha=0.85,
    edgecolor="white",
)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categorias, fontsize=8.5)
ax2.set_ylabel("Score relativo (mayor = mejor)")
ax2.set_title("Comparación de atributos", fontweight="bold", color=DARK)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
savefig("08-bow-vs-transformer.png")


# ─── 09. Dashboard resumen ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=FONDO)
fig.suptitle(
    "Tema 14: Fundamentos de NLP — Dashboard Resumen",
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
    wspace=0.4,
    left=0.06,
    right=0.97,
    top=0.90,
    bottom=0.06,
)

# Panel 1: TF-IDF vs BoW ventajas
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title("Cuándo usar TF-IDF vs BoW", fontsize=10, fontweight="bold", color=DARK)
items = [
    (
        "TF-IDF",
        VERDE,
        [
            "Penaliza palabras muy\ncomunes",
            "Resalta palabras\ndistintivas",
            "Mejor para búsqueda\ny clasificación",
        ],
    ),
    (
        "BoW",
        AZUL,
        [
            "Más simple e intuitivo",
            "Cuando el conteo\nimporta (detección spam)",
            "Datasets muy pequeños",
        ],
    ),
]
y = 0.85
for nombre, color, puntos in items:
    ax1.text(
        0.02,
        y,
        nombre,
        fontsize=9.5,
        fontweight="bold",
        color=color,
        transform=ax1.transAxes,
    )
    y -= 0.10
    for p in puntos:
        ax1.text(0.05, y, f"• {p}", fontsize=7.5, color=DARK, transform=ax1.transAxes)
        y -= 0.13
    y -= 0.04

# Panel 2: Herramientas NLP
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(FONDO)
ax2.axis("off")
ax2.set_title("Herramientas NLP en Python", fontsize=10, fontweight="bold", color=DARK)
herramientas = [
    ("NLTK", "Tokenizacion, stemming,\ntrees, corpus clasicos", AZUL),
    ("spaCy", "NLP industrial, lematizar,\nNER, POS, modelos ES", VERDE),
    ("sklearn", "TF-IDF, pipelines,\nclasificadores, eval", MORADO),
    ("HuggingFace", "Modelos preentrenados,\ntransformers, tokenizers", NARANJA),
    ("Gensim", "Word2Vec, LDA,\ntema modeling", CIAN),
]
y = 0.88
for nombre, desc, color in herramientas:
    rect = FancyBboxPatch(
        (0.02, y - 0.10),
        0.96,
        0.12,
        boxstyle="round,pad=0.01",
        facecolor=color + "22",
        edgecolor=color,
        linewidth=1.2,
        transform=ax2.transAxes,
    )
    ax2.add_patch(rect)
    ax2.text(
        0.06,
        y - 0.03,
        nombre,
        fontsize=8.5,
        fontweight="bold",
        color=color,
        transform=ax2.transAxes,
    )
    ax2.text(0.06, y - 0.08, desc, fontsize=7, color=DARK, transform=ax2.transAxes)
    y -= 0.16

# Panel 3: Mini-pipeline de sentimiento — accuracy por modelo
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(FONDO)
modelos_nlp = [
    "BoW+NB",
    "TF-IDF\n+LogReg",
    "TF-IDF\n+SVM",
    "TF-IDF\n+MLP",
    "BERT\nfine-tune",
]
accs = [0.78, 0.85, 0.87, 0.86, 0.93]
colors_bar = [GRIS, AZUL, VERDE, CIAN, MORADO]
bars = ax3.bar(modelos_nlp, accs, color=colors_bar, edgecolor="white", alpha=0.88)
ax3.set_ylabel("Accuracy (sentimiento)")
ax3.set_ylim(0.6, 1.0)
ax3.set_title(
    "Comparación de Modelos\n(Análisis de Sentimiento)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax3.grid(axis="y", alpha=0.3)
for bar, acc in zip(bars, accs):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.005,
        f"{acc:.0%}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

# Panel 4: Tokenización subword vs word
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor(FONDO)
ax4.axis("off")
ax4.set_title(
    "Tokenización: Word vs Subword (WordPiece/BPE)",
    fontsize=10,
    fontweight="bold",
    color=DARK,
)

tipos = [
    (
        "Word tokens",
        AZUL,
        ['"clasificacion"', '"aprendizaje"', '"desconocido"', "→ OOV ❌"],
    ),
    (
        "Subword (BPE)",
        VERDE,
        [
            '"clas" + "if" + "icacion"',
            '"apren" + "dizaje"',
            '"des" + "conoc" + "ido"',
            "→ no OOV ✓",
        ],
    ),
    (
        "Chars",
        NARANJA,
        ['"c"+"l"+"a"+"s"...', '"a"+"p"+"r"...', "nunca OOV", "→ secuencias largas ⚠️"],
    ),
]
col_x = [0.03, 0.36, 0.69]
for (titulo, color, rows), cx in zip(tipos, col_x):
    ax4.text(
        cx,
        0.92,
        titulo,
        fontsize=9.5,
        fontweight="bold",
        color=color,
        transform=ax4.transAxes,
    )
    for j, row in enumerate(rows):
        ax4.text(
            cx + 0.01,
            0.76 - j * 0.18,
            row,
            fontsize=8,
            color=DARK if j < 3 else color,
            transform=ax4.transAxes,
            family="monospace",
        )

# Panel 5: Curva TF vs IDF
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(FONDO)
N = 100
df_vals = np.linspace(1, N, 200)
idf_vals = np.log(N / df_vals) + 1
ax5.plot(df_vals, idf_vals, color=MORADO, lw=2.5)
ax5.fill_between(df_vals, idf_vals, alpha=0.15, color=MORADO)
ax5.axhline(y=1, color=GRIS, lw=1.2, ls="--", label="IDF mínimo")
ax5.set_xlabel("Frecuencia de documento (df)")
ax5.set_ylabel("IDF = log(N/df) + 1")
ax5.set_title(
    "Curva IDF: penaliza\npalabras comunes", fontsize=9, fontweight="bold", color=DARK
)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.annotate(
    "Palabras\nraras",
    xy=(5, idf_vals[5]),
    xytext=(20, 4.5),
    arrowprops=dict(arrowstyle="->", color=MORADO),
    fontsize=8,
    color=MORADO,
)
ax5.annotate(
    "Stopwords",
    xy=(95, idf_vals[-5]),
    xytext=(60, 1.5),
    arrowprops=dict(arrowstyle="->", color=ROJO),
    fontsize=8,
    color=ROJO,
)

savefig("09-dashboard.png")

print("\n✅ Todos los gráficos del tema 14 generados en:", OUT)
