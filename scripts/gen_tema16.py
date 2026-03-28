"""
gen_tema16.py — Genera 9 gráficos para el tema 16: Flujo de Trabajo en Kaggle y Competencias
Salida: public/ruta-aprendizaje-graficos/tema-16/
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

OUT = "public/ruta-aprendizaje-graficos/tema-16"
os.makedirs(OUT, exist_ok=True)

AZUL = "#2563EB"
VERDE = "#16A34A"
ROJO = "#DC2626"
NARANJA = "#EA580C"
MORADO = "#7C3AED"
GRIS = "#6B7280"
CIAN = "#0891B2"
ROSA = "#DB2777"
FONDO = "#F8FAFC"
DARK = "#1E293B"


def savefig(name):
    plt.savefig(f"{OUT}/{name}", dpi=130, bbox_inches="tight", facecolor=FONDO)
    plt.close()
    print(f"  ok {name}")


# ─── 01. Pipeline completo de competencia ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7), facecolor=FONDO)
ax.set_facecolor(FONDO)
ax.axis("off")
ax.set_title(
    "Pipeline Completo de una Competencia de ML/IA",
    fontsize=15,
    fontweight="bold",
    color=DARK,
    pad=10,
)

pasos = [
    (
        "1. Leer\nenunciado",
        AZUL,
        "#EFF6FF",
        "Metrica objetivo\nReglas de datos\nSubmission format",
    ),
    (
        "2. EDA\nExploratorio",
        VERDE,
        "#F0FDF4",
        "Distribuciones\nNulos y outliers\nCorrelaciones",
    ),
    (
        "3. Baseline\nRapido",
        CIAN,
        "#ECFEFF",
        "Modelo simple\nValidacion local\nPrimer submission",
    ),
    (
        "4. Feature\nEngineering",
        NARANJA,
        "#FFF7ED",
        "Transformaciones\nAggregaciones\nEncoding",
    ),
    (
        "5. Modelo\n+ Tuning",
        MORADO,
        "#FDF4FF",
        "Grid/Random search\nCross-validation\nEarly stopping",
    ),
    (
        "6. Ensemble\n+ Seleccion",
        ROSA,
        "#FDF2F8",
        "Blending/Stacking\nCV vs LB check\nSolucion final",
    ),
]

n = len(pasos)
W, H, gapx = 1.85, 1.5, 0.25
total_w = n * W + (n - 1) * gapx
x0 = (14 - total_w) / 2

for i, (titulo, color, bg, detalle) in enumerate(pasos):
    x = x0 + i * (W + gapx)
    y = 3.2
    rect = FancyBboxPatch(
        (x, y),
        W,
        H,
        boxstyle="round,pad=0.08",
        facecolor=bg,
        edgecolor=color,
        linewidth=2.2,
    )
    ax.add_patch(rect)
    ax.text(
        x + W / 2,
        y + H - 0.28,
        titulo,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
        multialignment="center",
    )
    ax.text(
        x + W / 2,
        y + 0.52,
        detalle,
        ha="center",
        va="center",
        fontsize=7.5,
        color=GRIS,
        multialignment="center",
    )
    if i < n - 1:
        ax.annotate(
            "",
            xy=(x + W + gapx, y + H / 2),
            xytext=(x + W, y + H / 2),
            arrowprops=dict(arrowstyle="->", color=GRIS, lw=2),
        )
    # numero de dia sugerido
    ax.text(
        x + W / 2,
        y - 0.35,
        f"Dia {i+1}",
        ha="center",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )

# Flecha de feedback (volver al paso 2 desde paso 5)
ax.annotate(
    "",
    xy=(x0 + 1 * (W + gapx) + W / 2, 3.0),
    xytext=(x0 + 4 * (W + gapx) + W / 2, 3.0),
    arrowprops=dict(
        arrowstyle="->", color=NARANJA, lw=1.8, connectionstyle="arc3,rad=0.4"
    ),
)
ax.text(
    (x0 + 2.5 * (W + gapx) + W / 2),
    2.55,
    "Ciclo iterativo: hipotesis → experimento → analisis",
    ha="center",
    fontsize=8.5,
    color=NARANJA,
    style="italic",
)

ax.set_xlim(0, 14)
ax.set_ylim(1.8, 5.5)
savefig("01-pipeline-competencia.png")


# ─── 02. EDA: distribución del target + importancia de features ───────────────
np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=FONDO)
fig.suptitle(
    "EDA Enfocado en Competencias: Target y Features",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

# Panel 1: distribución del target (regresion)
ax1 = axes[0]
ax1.set_facecolor(FONDO)
target = np.concatenate(
    [
        np.random.normal(45000, 12000, 600),
        np.random.normal(85000, 20000, 200),
        np.random.exponential(5000, 100),
    ]
)
ax1.hist(target, bins=40, color=AZUL, edgecolor="white", alpha=0.85)
ax1.axvline(
    np.mean(target), color=ROJO, lw=2.5, ls="--", label=f"Media={np.mean(target):.0f}"
)
ax1.axvline(
    np.median(target),
    color=VERDE,
    lw=2.5,
    ls="-.",
    label=f"Mediana={np.median(target):.0f}",
)
ax1.set_xlabel("Valor del target")
ax1.set_ylabel("Frecuencia")
ax1.set_title(
    "Distribucion del target\n(skew positivo -> transformar con log)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.3)
ax1.text(
    0.6,
    0.85,
    f"Skew: {float(np.mean([(t-np.mean(target))**3 for t in target])**(1/3) / np.std(target)):.2f}",
    transform=ax1.transAxes,
    fontsize=8.5,
    color=NARANJA,
)

# Panel 2: importancia de features (simulado)
ax2 = axes[1]
ax2.set_facecolor(FONDO)
features = [
    "ingreso_anual",
    "antiguedad",
    "n_productos",
    "edad",
    "ciudad_cat",
    "tipo_contrato",
    "historial_pago",
    "deuda_actual",
    "n_reclamos",
    "zona_geografica",
]
importancias = [0.22, 0.18, 0.15, 0.13, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02]
colors_imp = [VERDE if v > 0.10 else AZUL if v > 0.05 else GRIS for v in importancias]
bars = ax2.barh(
    range(len(features)), importancias, color=colors_imp, edgecolor="white", alpha=0.9
)
ax2.set_yticks(range(len(features)))
ax2.set_yticklabels(features, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("Importancia relativa")
ax2.set_title(
    "Feature Importance\n(XGBoost, fold promedio)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax2.axvline(x=0.05, color=ROJO, lw=1.5, ls="--", alpha=0.7, label="Umbral de corte")
ax2.legend(fontsize=8)
ax2.grid(axis="x", alpha=0.3)
for bar, v in zip(bars, importancias):
    ax2.text(
        v + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.2f}",
        va="center",
        fontsize=8,
    )

# Panel 3: valores nulos por feature
ax3 = axes[2]
ax3.set_facecolor(FONDO)
features_null = [
    "ingreso_anual",
    "historial_pago",
    "deuda_actual",
    "n_reclamos",
    "ciudad_cat",
    "tipo_contrato",
]
pct_null = [0.0, 0.03, 0.12, 0.25, 0.38, 0.02]
AMARILLO = "#D97706"
colors_null = [VERDE if v < 0.05 else AMARILLO if v < 0.20 else ROJO for v in pct_null]
bars3 = ax3.barh(
    features_null, pct_null, color=colors_null, edgecolor="white", alpha=0.9
)
ax3.axvline(x=0.20, color=ROJO, lw=2, ls="--", alpha=0.8, label="Umbral 20%")
ax3.axvline(x=0.05, color=NARANJA, lw=1.5, ls=":", alpha=0.8, label="Umbral 5%")
ax3.set_xlabel("Porcentaje de nulos")
ax3.set_title(
    "Valores Nulos por Feature\n(orienta estrategia de imputacion)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax3.legend(fontsize=8)
ax3.grid(axis="x", alpha=0.3)
for bar, v in zip(bars3, pct_null):
    ax3.text(
        v + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.0%}",
        va="center",
        fontsize=9,
    )

plt.tight_layout()
savefig("02-eda-target-features.png")


# ─── 03. Validación cruzada vs Leaderboard ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=FONDO)
fig.suptitle(
    "Estrategia de Validacion: Evitar que el LB Publico te Engane",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

# Panel izq: correlacion CV vs LB publico
ax1 = axes[0]
ax1.set_facecolor(FONDO)
cv_scores = np.array(
    [0.832, 0.840, 0.845, 0.838, 0.851, 0.849, 0.856, 0.843, 0.862, 0.858, 0.865, 0.871]
)
lb_pub = cv_scores + np.random.normal(0, 0.006, len(cv_scores))
lb_priv = cv_scores + np.random.normal(-0.002, 0.004, len(cv_scores))

ax1.scatter(cv_scores, lb_pub, color=AZUL, s=80, zorder=5, label="LB Publico (20%)")
ax1.scatter(
    cv_scores,
    lb_priv,
    color=VERDE,
    s=80,
    zorder=5,
    marker="^",
    label="LB Privado (80%)",
)
# Linea de identidad
lims = [
    min(cv_scores.min(), lb_pub.min()) - 0.005,
    max(cv_scores.max(), lb_pub.max()) + 0.005,
]
ax1.plot(lims, lims, "k--", lw=1.5, alpha=0.5, label="CV = LB ideal")

# Correlaciones
from numpy.polynomial.polynomial import polyfit

c_pub = np.corrcoef(cv_scores, lb_pub)[0, 1]
c_priv = np.corrcoef(cv_scores, lb_priv)[0, 1]
ax1.text(
    0.05,
    0.90,
    f"Corr CV-LBpub:  {c_pub:.3f}",
    transform=ax1.transAxes,
    fontsize=9,
    color=AZUL,
)
ax1.text(
    0.05,
    0.82,
    f"Corr CV-LBpriv: {c_priv:.3f}",
    transform=ax1.transAxes,
    fontsize=9,
    color=VERDE,
)
ax1.set_xlabel("CV Score (local)")
ax1.set_ylabel("Leaderboard Score")
ax1.set_title(
    "CV vs Leaderboard: confiar en CV si\ncorrelacion es alta",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel der: degradacion del LB publico por overfit
ax2 = axes[1]
ax2.set_facecolor(FONDO)
n_submits = np.arange(1, 21)
# Equipo A: usa CV riguroso, mejora consistente
lb_A = 0.820 + 0.002 * np.log(n_submits) + np.random.normal(0, 0.003, 20)
# Equipo B: hace overfit al LB publico
lb_B_pub = 0.820 + 0.004 * np.log(n_submits) + np.random.normal(0, 0.004, 20)
lb_B_priv = (
    0.820
    + 0.001 * np.log(n_submits)
    - 0.008 * (n_submits / 20) ** 2
    + np.random.normal(0, 0.003, 20)
)

ax2.plot(
    n_submits, lb_A, "o-", color=VERDE, lw=2.5, ms=5, label="Equipo A: CV rigoroso"
)
ax2.plot(
    n_submits,
    lb_B_pub,
    "s-",
    color=AZUL,
    lw=2.5,
    ms=5,
    label="Equipo B: LB pub (optimista)",
)
ax2.plot(
    n_submits,
    lb_B_priv,
    "^--",
    color=ROJO,
    lw=2.5,
    ms=5,
    label="Equipo B: LB priv (realidad)",
)
ax2.axvline(x=15, color=GRIS, lw=1.5, ls=":", alpha=0.7)
ax2.text(15.3, lb_B_pub.min(), "Submission\nlimit", fontsize=8, color=GRIS)
ax2.set_xlabel("Numero de submissions")
ax2.set_ylabel("Score")
ax2.set_title(
    "Peligro de hacer overfit al\nLeaderboard Publico",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
savefig("03-cv-vs-leaderboard.png")


# ─── 04. Tracking de experimentos ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=FONDO)
fig.suptitle(
    "Tracking de Experimentos: Sin Registro No Hay Mejora Reproducible",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

# Panel izq: tabla de experimentos simulada
ax1 = axes[0]
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title(
    "Tabla de experimentos (formato recomendado)", fontweight="bold", color=DARK, pad=8
)

headers = ["Exp", "Modelo", "Features", "CV F1", "LB pub", "Nota"]
rows = [
    ["#01", "LogReg", "TF-IDF base", "0.782", "0.778", "Baseline"],
    ["#02", "XGBoost", "TF-IDF base", "0.811", "0.808", "+3pts vs baseline"],
    ["#03", "XGBoost", "+ ngrams 1-3", "0.819", "0.815", "+bigramas ayuda"],
    ["#04", "XGBoost", "+ embeddings", "0.824", "0.820", "mejora leve"],
    ["#05", "LGBM", "idem exp04", "0.827", "0.822", "LGBM > XGB aqui"],
    ["#06", "LGBM", "sin stopwords", "0.821", "0.818", "peor, volver atras"],
    ["#07", "LGBM", "tuning lr=0.05", "0.831", "0.826", "mejor hasta ahora"],
    ["#08", "Blend", "07+BERT base", "0.843", "0.839", "ensemble gana"],
    ["#09", "Blend", "07+BERT+LogReg", "0.846", "0.841", "mejora marginal"],
    ["#10", "STACK", "meta=LogReg cv=5", "0.849", "0.843", "submit final"],
]

col_widths = [0.07, 0.12, 0.28, 0.10, 0.10, 0.28]
col_x = [0.01]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w)

for j, (h, x) in enumerate(zip(headers, col_x)):
    ax1.text(
        x,
        0.96,
        h,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color=AZUL,
        transform=ax1.transAxes,
    )

for i, row in enumerate(rows):
    y = 0.88 - i * 0.088
    bg = "#EFF6FF" if i % 2 == 0 else FONDO
    rect = FancyBboxPatch(
        (0, y - 0.038),
        1.0,
        0.075,
        boxstyle="round,pad=0.005",
        facecolor=bg,
        edgecolor="none",
        transform=ax1.transAxes,
    )
    ax1.add_patch(rect)
    # Highlight mejor resultado
    if row[0] == "#10":
        rect2 = FancyBboxPatch(
            (0, y - 0.038),
            1.0,
            0.075,
            boxstyle="round,pad=0.005",
            facecolor=VERDE + "33",
            edgecolor=VERDE,
            linewidth=1.2,
            transform=ax1.transAxes,
        )
        ax1.add_patch(rect2)
    for val, x, col in zip(row, col_x, col_widths):
        color = VERDE if val in ["0.849", "0.843"] else ROJO if val == "0.821" else DARK
        ax1.text(
            x,
            y,
            val,
            ha="left",
            va="center",
            fontsize=7.5,
            color=color,
            transform=ax1.transAxes,
        )

# Panel der: grafico de progresion
ax2 = axes[1]
ax2.set_facecolor(FONDO)
exps = [f"#{i+1:02d}" for i in range(10)]
cv_vals = [0.782, 0.811, 0.819, 0.824, 0.827, 0.821, 0.831, 0.843, 0.846, 0.849]
lb_vals = [0.778, 0.808, 0.815, 0.820, 0.822, 0.818, 0.826, 0.839, 0.841, 0.843]
x_pos = np.arange(len(exps))

ax2.plot(x_pos, cv_vals, "o-", color=AZUL, lw=2.5, ms=6, label="CV Score (local)")
ax2.plot(x_pos, lb_vals, "s--", color=VERDE, lw=2.5, ms=6, label="LB Publico")
ax2.fill_between(x_pos, cv_vals, lb_vals, alpha=0.08, color=GRIS)
# Marcar el paso atras (exp 06)
ax2.scatter([5], [cv_vals[5]], color=ROJO, s=120, zorder=6)
ax2.annotate(
    "Paso atras\n(stopwords off)",
    xy=(5, cv_vals[5]),
    xytext=(4, 0.815),
    arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.5),
    fontsize=8,
    color=ROJO,
)
# Marcar ensemble
ax2.axvspan(7, 9, alpha=0.08, color=MORADO)
ax2.text(
    8, 0.800, "Ensemble", ha="center", fontsize=8.5, color=MORADO, fontweight="bold"
)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(exps, fontsize=9)
ax2.set_ylabel("F1-macro")
ax2.set_ylim(0.77, 0.86)
ax2.set_title(
    "Progresion de experimentos\n(registrar todo, incluso los retrocesos)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
savefig("04-tracking-experimentos.png")


# ─── 05. Estrategias de Ensemble ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=FONDO)
fig.suptitle(
    "Estrategias de Ensemble: Combinar Modelos para Ganar Robustez",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

# Panel izq: diagrama de ensemble
ax1 = axes[0]
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title("Blending vs Stacking", fontweight="bold", color=DARK, pad=8)

# Modelos base
modelos_b = [
    ("XGBoost\nF1=0.831", NARANJA, 0.72),
    ("LGBM\nF1=0.828", VERDE, 0.55),
    ("BERT\nF1=0.839", MORADO, 0.38),
    ("LogReg\nF1=0.782", AZUL, 0.21),
]

for nombre, col, y in modelos_b:
    rect = FancyBboxPatch(
        (0.05, y - 0.05),
        0.25,
        0.10,
        boxstyle="round,pad=0.02",
        facecolor=col + "33",
        edgecolor=col,
        linewidth=1.8,
        transform=ax1.transAxes,
    )
    ax1.add_patch(rect)
    ax1.text(
        0.175,
        y,
        nombre,
        ha="center",
        va="center",
        fontsize=8.5,
        color=col,
        fontweight="bold",
        transform=ax1.transAxes,
        multialignment="center",
    )
    # flecha al blending
    ax1.annotate(
        "",
        xy=(0.50, 0.47),
        xytext=(0.30, y),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.7),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    # flecha al stacking
    ax1.annotate(
        "",
        xy=(0.50, 0.17),
        xytext=(0.30, y),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )

# Blending box
rect_bl = FancyBboxPatch(
    (0.50, 0.38),
    0.24,
    0.16,
    boxstyle="round,pad=0.02",
    facecolor="#FFF7ED",
    edgecolor=NARANJA,
    linewidth=2,
    transform=ax1.transAxes,
)
ax1.add_patch(rect_bl)
ax1.text(
    0.62,
    0.46,
    "BLENDING\npeso*pred_i",
    ha="center",
    va="center",
    fontsize=8.5,
    color=NARANJA,
    fontweight="bold",
    transform=ax1.transAxes,
    multialignment="center",
)

# Stacking box
rect_st = FancyBboxPatch(
    (0.50, 0.08),
    0.24,
    0.16,
    boxstyle="round,pad=0.02",
    facecolor="#EFF6FF",
    edgecolor=AZUL,
    linewidth=2,
    transform=ax1.transAxes,
)
ax1.add_patch(rect_st)
ax1.text(
    0.62,
    0.16,
    "STACKING\nmeta-modelo\n(LogReg/LR)",
    ha="center",
    va="center",
    fontsize=8.5,
    color=AZUL,
    fontweight="bold",
    transform=ax1.transAxes,
    multialignment="center",
)

# Outputs
for label, col, y_box in [("F1=0.843", NARANJA, 0.46), ("F1=0.849", AZUL, 0.16)]:
    ax1.annotate(
        "",
        xy=(0.88, y_box),
        xytext=(0.74, y_box),
        arrowprops=dict(arrowstyle="->", color=col, lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    rect_out = FancyBboxPatch(
        (0.88, y_box - 0.06),
        0.10,
        0.12,
        boxstyle="round,pad=0.02",
        facecolor=col + "22",
        edgecolor=col,
        linewidth=1.5,
        transform=ax1.transAxes,
    )
    ax1.add_patch(rect_out)
    ax1.text(
        0.93,
        y_box,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color=col,
        fontweight="bold",
        transform=ax1.transAxes,
    )

# Diferencia clave
ax1.text(
    0.5,
    0.95,
    "Blending: promedio ponderado de predicciones OOF",
    ha="center",
    fontsize=8,
    color=NARANJA,
    style="italic",
    transform=ax1.transAxes,
)
ax1.text(
    0.5,
    0.89,
    "Stacking: meta-modelo aprende como combinar predicciones OOF",
    ha="center",
    fontsize=8,
    color=AZUL,
    style="italic",
    transform=ax1.transAxes,
)

# Panel der: mejora de ensemble con diversidad
ax2 = axes[1]
ax2.set_facecolor(FONDO)
correlaciones = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
# Mejora teorica del ensemble vs modelo individual (2 modelos iguales de F1=0.85)
base_f1 = 0.85
mejora = base_f1 * (1 + (1 - correlaciones) * 0.10)  # simplificado
ax2.plot(correlaciones, mejora, "o-", color=MORADO, lw=2.5, ms=6)
ax2.axhline(
    y=base_f1, color=GRIS, lw=2, ls="--", label=f"Modelo individual (F1={base_f1})"
)
ax2.fill_between(
    correlaciones,
    base_f1,
    mejora,
    alpha=0.12,
    color=MORADO,
    label="Ganancia del ensemble",
)
ax2.set_xlabel("Correlacion entre modelos (+ alta = menos diverso)")
ax2.set_ylabel("F1 del ensemble")
ax2.set_title(
    "Ensemble gana mas cuando los\nmodelos son DIVERSOS (baja correlacion)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Anotaciones
ax2.annotate(
    "Muy\ndiversos",
    xy=(0.1, mejora[0]),
    xytext=(0.18, mejora[0] + 0.006),
    arrowprops=dict(arrowstyle="->", color=VERDE, lw=1.5),
    fontsize=8.5,
    color=VERDE,
)
ax2.annotate(
    "Casi iguales\n(no ayuda)",
    xy=(0.95, mejora[-1]),
    xytext=(0.75, mejora[-1] + 0.008),
    arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.5),
    fontsize=8.5,
    color=ROJO,
)

plt.tight_layout()
savefig("05-estrategias-ensemble.png")


# ─── 06. Data Leakage: tipos y detección ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor=FONDO)
fig.suptitle(
    "Data Leakage: El Error mas Costoso en Competencias",
    fontsize=14,
    fontweight="bold",
    color=DARK,
)

# Panel izq: tipos de leakage
ax1 = axes[0]
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title("Tipos de Data Leakage", fontweight="bold", color=DARK, pad=8)

tipos = [
    (
        "Target Leakage",
        ROJO,
        "Feature creada DESPUES del evento target.\nEj: 'recibio_tratamiento' para predecir\n'se_enfermo' (el tratamiento implica enfermedad)",
        "Verificar causalidad temporal",
    ),
    (
        "Train-Test\nContaminacion",
        NARANJA,
        "Estadisticas del test set se filtran al train.\nEj: normalizar con mean/std calculada\nsobre train + test juntos",
        "Ajustar scaler SOLO en train",
    ),
    (
        "Leakage de\nID/Timestamp",
        MORADO,
        "El ID o timestamp contiene informacion\ndel target. Ej: ID asignado en orden\nde resultado positivo",
        "Analizar distribucion de IDs",
    ),
    (
        "Group Leakage",
        AZUL,
        "El mismo 'sujeto' aparece en train y test.\nEj: cliente con multiples transacciones\nsplit al azar (no por cliente)",
        "Usar GroupKFold por entidad",
    ),
]

y = 0.95
for titulo, col, desc, solucion in tipos:
    rect = FancyBboxPatch(
        (0.01, y - 0.22),
        0.98,
        0.21,
        boxstyle="round,pad=0.02",
        facecolor=col + "18",
        edgecolor=col,
        linewidth=1.8,
        transform=ax1.transAxes,
    )
    ax1.add_patch(rect)
    ax1.text(
        0.03,
        y - 0.02,
        titulo,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=col,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.03,
        y - 0.08,
        desc,
        ha="left",
        va="top",
        fontsize=7.5,
        color=DARK,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.03,
        y - 0.17,
        f"Solucion: {solucion}",
        ha="left",
        va="top",
        fontsize=7.5,
        color=VERDE,
        style="italic",
        transform=ax1.transAxes,
    )
    y -= 0.25

# Panel der: señales de leakage
ax2 = axes[1]
ax2.set_facecolor(FONDO)
ax2.set_title(
    "Señales de Alerta de Posible Leakage", fontweight="bold", color=DARK, pad=8
)

# Simular: modelo con leakage vs sin leakage
np.random.seed(55)
# Con leakage: train muy alto, test mucho mas bajo
cv_splits = np.arange(1, 6)
with_leak_train = np.array([0.98, 0.97, 0.99, 0.98, 0.97])
with_leak_val = np.array([0.72, 0.69, 0.74, 0.71, 0.73])
no_leak_train = np.array([0.88, 0.87, 0.89, 0.88, 0.87])
no_leak_val = np.array([0.84, 0.83, 0.85, 0.84, 0.83])

x = np.arange(len(cv_splits))
w = 0.2
ax2.bar(
    x - 1.5 * w,
    with_leak_train,
    w,
    color=ROJO,
    alpha=0.8,
    edgecolor="white",
    label="Con leakage (train)",
)
ax2.bar(
    x - 0.5 * w,
    with_leak_val,
    w,
    color=NARANJA,
    alpha=0.8,
    edgecolor="white",
    label="Con leakage (val)",
)
ax2.bar(
    x + 0.5 * w,
    no_leak_train,
    w,
    color=AZUL,
    alpha=0.8,
    edgecolor="white",
    label="Sin leakage (train)",
)
ax2.bar(
    x + 1.5 * w,
    no_leak_val,
    w,
    color=VERDE,
    alpha=0.8,
    edgecolor="white",
    label="Sin leakage (val)",
)

ax2.set_xticks(x)
ax2.set_xticklabels([f"Fold {i+1}" for i in range(5)])
ax2.set_ylabel("Score")
ax2.set_ylim(0.55, 1.05)
ax2.legend(fontsize=7.5, ncol=2)
ax2.grid(axis="y", alpha=0.3)

# Anotacion de la diferencia con leakage
ax2.annotate(
    "",
    xy=(0 - 0.5 * w, with_leak_val[0] + 0.01),
    xytext=(0 - 1.5 * w, with_leak_train[0] - 0.01),
    arrowprops=dict(arrowstyle="<->", color=ROJO, lw=2),
)
ax2.text(
    -0.1,
    0.86,
    "Gap\nsospechoso",
    ha="center",
    fontsize=8.5,
    color=ROJO,
    fontweight="bold",
)
ax2.text(
    0.5,
    0.58,
    "Señal de leakage: Train >> Val en TODOS los folds",
    ha="center",
    fontsize=9,
    color=ROJO,
    style="italic",
    transform=ax2.transAxes,
)

plt.tight_layout()
savefig("06-data-leakage.png")


# ─── 07. Feature Engineering: impacto de transformaciones ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=FONDO)
fig.suptitle(
    "Feature Engineering: Transformaciones Clave para Tabular Data",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

np.random.seed(77)

# Panel 1: log-transform de target con skew
ax1 = axes[0, 0]
ax1.set_facecolor(FONDO)
raw = np.random.exponential(scale=20000, size=1000)
logt = np.log1p(raw)
ax1.hist(raw, bins=40, color=ROJO, alpha=0.7, label="Original (skew)", density=True)
ax1_twin = ax1.twinx()
ax1_twin.hist(logt, bins=40, color=VERDE, alpha=0.7, label="log1p(x)", density=True)
ax1.set_xlabel("Valor")
ax1.set_ylabel("Densidad (original)", color=ROJO)
ax1_twin.set_ylabel("Densidad (log)", color=VERDE)
ax1.set_title(
    "Log-transform: normaliza targets\ny features con skew positivo",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# Panel 2: encoding categorico
ax2 = axes[0, 1]
ax2.set_facecolor(FONDO)
categorias = ["ciudad_A", "ciudad_B", "ciudad_C", "ciudad_D", "ciudad_E"]
n_cat = len(categorias)
# OHE genera n columnas, target encoding genera 1
ohe_dims = [n_cat] * n_cat
target_enc_vals = [0.45, 0.72, 0.31, 0.58, 0.67]  # mean target por categoria
count_vals = [320, 180, 410, 95, 240]

x = np.arange(n_cat)
ax2.bar(
    x - 0.2,
    target_enc_vals,
    0.35,
    color=VERDE,
    alpha=0.85,
    edgecolor="white",
    label="Target Encoding (1 col)",
)
ax2.scatter(
    x,
    [v / max(count_vals) for v in count_vals],
    color=MORADO,
    s=80,
    zorder=5,
    label="Frecuencia relativa",
)
ax2.set_xticks(x)
ax2.set_xticklabels(categorias, rotation=20, ha="right", fontsize=9)
ax2.set_ylabel("Target mean / Frecuencia")
ax2.set_title(
    "Encoding Categorico: Target Encoding\nvs One-Hot (muchas columnas)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

# Panel 3: features de interaccion
ax3 = axes[1, 0]
ax3.set_facecolor(FONDO)
feat_names = ["edad", "ingreso", "deuda", "edad*ingreso", "deuda/ingreso", "edad^2"]
importancia_antes = [0.08, 0.15, 0.12, 0.00, 0.00, 0.00]
importancia_despues = [0.07, 0.14, 0.11, 0.18, 0.21, 0.09]

x = np.arange(len(feat_names))
ax3.bar(
    x - 0.2,
    importancia_antes,
    0.35,
    color=GRIS,
    alpha=0.8,
    edgecolor="white",
    label="Sin interacciones",
)
ax3.bar(
    x + 0.2,
    importancia_despues,
    0.35,
    color=VERDE,
    alpha=0.8,
    edgecolor="white",
    label="Con interacciones",
)
ax3.set_xticks(x)
ax3.set_xticklabels(feat_names, rotation=20, ha="right", fontsize=9)
ax3.set_ylabel("Importancia")
ax3.set_title(
    "Features de Interaccion: revelan\nrelaciones no lineales",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

# Panel 4: cross-validation estrategias
ax4 = axes[1, 1]
ax4.set_facecolor(FONDO)
ax4.axis("off")
ax4.set_title(
    "Tipos de Cross-Validation segun el problema",
    fontsize=9,
    fontweight="bold",
    color=DARK,
    pad=8,
)

tipos_cv = [
    ("KFold", AZUL, "Datos i.i.d., clasificacion/regresion general"),
    ("StratifiedKFold", VERDE, "Clases desbalanceadas (preserva % por clase)"),
    (
        "GroupKFold",
        MORADO,
        "Un 'grupo' (cliente, paciente) no debe aparecer\nen train Y val al mismo tiempo",
    ),
    ("TimeSeriesSplit", NARANJA, "Series temporales: val siempre DESPUES del train"),
    ("RepeatedKFold", CIAN, "Alta varianza: repite CV N veces con distinto seed"),
]
y = 0.95
for nombre, col, desc in tipos_cv:
    rect = FancyBboxPatch(
        (0.01, y - 0.14),
        0.98,
        0.13,
        boxstyle="round,pad=0.01",
        facecolor=col + "20",
        edgecolor=col,
        linewidth=1.5,
        transform=ax4.transAxes,
    )
    ax4.add_patch(rect)
    ax4.text(
        0.04,
        y - 0.035,
        nombre,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=col,
        transform=ax4.transAxes,
    )
    ax4.text(
        0.04,
        y - 0.09,
        desc,
        ha="left",
        va="center",
        fontsize=7.5,
        color=DARK,
        transform=ax4.transAxes,
    )
    y -= 0.19

plt.tight_layout()
savefig("07-feature-engineering.png")


# ─── 08. Hiperparámetro tuning: Random vs Grid vs Bayesiano ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=FONDO)
fig.suptitle(
    "Estrategias de Hyperparameter Tuning: Eficiencia vs Exhaustividad",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

np.random.seed(42)


# Funcion de objetivo (2D simulada)
def score_fn(lr, max_depth):
    return (
        0.9
        - 0.5 * (lr - 0.1) ** 2
        - 0.3 * (max_depth - 6) ** 2 / 36
        + np.random.normal(0, 0.005)
    )


lr_range = np.linspace(0.01, 0.5, 100)
md_range = np.linspace(3, 12, 100)
LR_g, MD_g = np.meshgrid(lr_range, md_range)
Z = 0.9 - 0.5 * (LR_g - 0.1) ** 2 - 0.3 * (MD_g - 6) ** 2 / 36

for ax, (titulo, color, desc) in zip(
    axes,
    [
        (
            "Grid Search\n(exhaustivo)",
            AZUL,
            "Evalua todos los\ncombinaciones\n(costoso pero completo)",
        ),
        (
            "Random Search\n(aleatorio)",
            VERDE,
            "Muestrea puntos\nal azar (mejor\nen espacios grandes)",
        ),
        (
            "Optuna / Bayesiano\n(inteligente)",
            MORADO,
            "Aprende del pasado\npara proponer mejores\npuntos (optimo)",
        ),
    ],
):
    ax.set_facecolor(FONDO)
    im = ax.contourf(LR_g, MD_g, Z, levels=20, cmap="YlOrRd", alpha=0.7)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Max Depth")
    ax.set_title(titulo, fontweight="bold", color=color, pad=6)

    if titulo.startswith("Grid"):
        # Grid: puntos en cuadricula
        lrs_g = np.linspace(0.01, 0.5, 6)
        mds_g = np.linspace(3, 12, 5)
        LR_pts, MD_pts = np.meshgrid(lrs_g, mds_g)
        ax.scatter(
            LR_pts.flatten(), MD_pts.flatten(), color=AZUL, s=35, zorder=5, alpha=0.9
        )
        ax.text(
            0.05,
            0.05,
            f"30 eval",
            transform=ax.transAxes,
            fontsize=9,
            color=AZUL,
            fontweight="bold",
        )

    elif titulo.startswith("Random"):
        # Random: puntos dispersos
        lrs_r = np.random.uniform(0.01, 0.5, 30)
        mds_r = np.random.uniform(3, 12, 30)
        ax.scatter(lrs_r, mds_r, color=VERDE, s=35, zorder=5, alpha=0.9)
        ax.text(
            0.05,
            0.05,
            f"30 eval",
            transform=ax.transAxes,
            fontsize=9,
            color=VERDE,
            fontweight="bold",
        )

    else:
        # Optuna: concentra en zona de alto score
        lrs_o = np.concatenate(
            [np.random.uniform(0.01, 0.5, 10), np.random.normal(0.10, 0.05, 20)]
        )
        mds_o = np.concatenate(
            [np.random.uniform(3, 12, 10), np.random.normal(6, 1.5, 20)]
        )
        lrs_o = np.clip(lrs_o, 0.01, 0.5)
        mds_o = np.clip(mds_o, 3, 12)
        scores_o = [score_fn(lr, md) for lr, md in zip(lrs_o, mds_o)]
        ax.scatter(
            lrs_o,
            mds_o,
            c=scores_o,
            cmap="RdYlGn",
            s=50,
            zorder=5,
            vmin=0.7,
            vmax=0.92,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.text(
            0.05,
            0.05,
            "30 eval\n(mejor zona)",
            transform=ax.transAxes,
            fontsize=9,
            color=MORADO,
            fontweight="bold",
        )

    plt.colorbar(im, ax=ax, label="Score", shrink=0.85)
    # Marcar optimo
    ax.scatter(
        [0.10],
        [6],
        color="white",
        s=150,
        zorder=6,
        edgecolors="black",
        linewidth=2,
        marker="*",
    )
    ax.text(0.12, 6.3, "optimo", fontsize=8.5, color="black", fontweight="bold")

plt.tight_layout()
savefig("08-hyperparameter-tuning.png")


# ─── 09. Dashboard resumen ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=FONDO)
fig.suptitle(
    "Tema 16: Flujo de Trabajo en Kaggle y Competencias — Dashboard",
    fontsize=15,
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
    top=0.91,
    bottom=0.06,
)

# Panel 1: Checklist de competencia
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title("Checklist de Competencia", fontsize=10, fontweight="bold", color=DARK)
items_check = [
    ("Leer enunciado completo", VERDE, True),
    ("Entender la metrica", VERDE, True),
    ("EDA basico hecho", VERDE, True),
    ("Baseline en < 1 hora", VERDE, True),
    ("CV local definido", VERDE, True),
    ("Tracking de experimentos", VERDE, True),
    ("Sin data leakage", VERDE, True),
    ("Submission validada", VERDE, True),
    ("Ensemble final preparado", VERDE, True),
    ("Dos submissions distintas", NARANJA, False),
]
y = 0.95
for item, col, done in items_check:
    mark = "[ok]" if done else "[ ]"
    ax1.text(
        0.02,
        y,
        mark,
        fontsize=9,
        color=col if done else GRIS,
        fontweight="bold",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.15,
        y,
        item,
        fontsize=8.5,
        color=DARK if done else GRIS,
        transform=ax1.transAxes,
    )
    y -= 0.095

# Panel 2: Metrica segun tipo de problema
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(FONDO)
ax2.axis("off")
ax2.set_title(
    "Metrica segun Tipo de Problema", fontsize=10, fontweight="bold", color=DARK
)
metricas_tabla = [
    ("Clasificacion binaria", "ROC-AUC, F1, Logloss"),
    ("Clasif. multiclase", "F1-macro, Logloss"),
    ("Regresion", "RMSE, MAE, R2"),
    ("Clases desbalanceadas", "F1-macro, PR-AUC"),
    ("Ranking", "NDCG, MAP"),
    ("Segmentacion", "mIoU, Dice"),
    ("Deteccion objetos", "mAP@50, mAP@75"),
    ("NLP generacion", "BLEU, ROUGE, BERTScore"),
]
y = 0.95
for tipo, metrica in metricas_tabla:
    ax2.text(
        0.01,
        y,
        tipo,
        fontsize=8.5,
        color=AZUL,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.01,
        y - 0.06,
        f"  -> {metrica}",
        fontsize=7.5,
        color=DARK,
        transform=ax2.transAxes,
    )
    y -= 0.125

# Panel 3: Curva de progresion tipica
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(FONDO)
dias = np.arange(1, 8)
progresion = np.array([0.78, 0.81, 0.83, 0.84, 0.845, 0.850, 0.852])
ax3.plot(dias, progresion, "o-", color=AZUL, lw=2.5, ms=7)
for d, p, label in [
    (1, 0.78, "Baseline"),
    (3, 0.83, "Feature eng."),
    (5, 0.845, "Tuning"),
    (6, 0.850, "Ensemble"),
    (7, 0.852, "Submit final"),
]:
    ax3.annotate(
        label, xy=(d, p), xytext=(d + 0.1, p + 0.003), fontsize=7.5, color=AZUL
    )
ax3.fill_between(dias, progresion, progresion[0], alpha=0.1, color=AZUL)
ax3.set_xlabel("Dia de la competencia")
ax3.set_ylabel("CV Score")
ax3.set_title(
    "Progresion tipica\nen una competencia de 7 dias",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax3.set_ylim(0.75, 0.87)
ax3.grid(True, alpha=0.3)

# Panel 4: Comparacion de estrategias de submit
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor(FONDO)
estrategias = [
    "Mejor CV local",
    "Mejor LB pub",
    "Ensemble A\n(diverso)",
    "Ensemble B\n(homogeneo)",
    "Seleccion\ncautela",
]
cv_est = [0.851, 0.846, 0.855, 0.852, 0.853]
lb_pub = [0.848, 0.851, 0.852, 0.851, 0.850]
lb_priv = [0.849, 0.839, 0.854, 0.848, 0.851]  # la realidad

x = np.arange(len(estrategias))
w = 0.25
ax4.bar(x - w, cv_est, w, color=AZUL, alpha=0.85, edgecolor="white", label="CV local")
ax4.bar(x, lb_pub, w, color=VERDE, alpha=0.85, edgecolor="white", label="LB Publico")
ax4.bar(
    x + w,
    lb_priv,
    w,
    color=MORADO,
    alpha=0.85,
    edgecolor="white",
    label="LB Privado (real)",
)
ax4.set_xticks(x)
ax4.set_xticklabels(estrategias, fontsize=9)
ax4.set_ylabel("Score")
ax4.set_ylim(0.83, 0.86)
ax4.set_title(
    "Comparacion de estrategias de submission: confiar en CV > LB publico",
    fontweight="bold",
    fontsize=10,
    color=DARK,
)
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.3)
# Destacar ganador
ax4.annotate(
    "Ganador\nreal",
    xy=(2 + w, lb_priv[2]),
    xytext=(2 + w, 0.858),
    ha="center",
    arrowprops=dict(arrowstyle="->", color=MORADO, lw=2),
    fontsize=9,
    color=MORADO,
    fontweight="bold",
)

# Panel 5: Tiempo recomendado por fase
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(FONDO)
fases = [
    "EDA\n(10%)",
    "Baseline\n(10%)",
    "Feature\nEng.(30%)",
    "Modelo\n+Tuning(30%)",
    "Ensemble\n(20%)",
]
tiempos = [10, 10, 30, 30, 20]
colores_pie = [AZUL, VERDE, NARANJA, MORADO, CIAN]
wedges, texts, autotexts = ax5.pie(
    tiempos,
    labels=fases,
    colors=colores_pie,
    autopct="%1.0f%%",
    pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
for t in texts:
    t.set_fontsize(8)
ax5.set_title("Distribucion\ndel tiempo", fontsize=9, fontweight="bold", color=DARK)

savefig("09-dashboard.png")

print("\nTodos los graficos del tema 16 generados en:", OUT)
