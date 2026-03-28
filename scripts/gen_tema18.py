"""
gen_tema18.py — Genera 9 gráficos para el tema 18: Series Temporales y Datos Secuenciales
Salida: public/ruta-aprendizaje-graficos/tema-18/
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "public/ruta-aprendizaje-graficos/tema-18"
os.makedirs(OUT, exist_ok=True)

AZUL = "#2563EB"
VERDE = "#16A34A"
ROJO = "#DC2626"
NARANJA = "#EA580C"
MORADO = "#7C3AED"
GRIS = "#6B7280"
CIAN = "#0891B2"
ROSA = "#DB2777"
AMARILLO = "#D97706"
FONDO = "#F8FAFC"
DARK = "#1E293B"


def savefig(name):
    plt.savefig(f"{OUT}/{name}", dpi=130, bbox_inches="tight", facecolor=FONDO)
    plt.close()
    print(f"  ok {name}")


# ─── 01. Descomposición de una serie temporal ─────────────────────────────────
np.random.seed(42)
t = np.arange(0, 365 * 3)  # 3 años de datos diarios

tendencia = 100 + 0.08 * t
estacionalidad = 15 * np.sin(2 * np.pi * t / 365) + 7 * np.sin(2 * np.pi * t / 7)
ruido = np.random.normal(0, 4, len(t))
serie = tendencia + estacionalidad + ruido

fig, axes = plt.subplots(4, 1, figsize=(13, 10), facecolor=FONDO, sharex=True)
fig.suptitle(
    "Descomposicion de una Serie Temporal: Tendencia + Estacionalidad + Ruido",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

components = [
    (serie, AZUL, "Serie original = T + S + R"),
    (tendencia, NARANJA, "Tendencia (T): direccion a largo plazo"),
    (estacionalidad, MORADO, "Estacionalidad (S): patron repetitivo"),
    (ruido, GRIS, "Residuo / Ruido (R): lo no explicado"),
]
for ax, (data, color, titulo) in zip(axes, components):
    ax.set_facecolor(FONDO)
    ax.plot(t, data, color=color, lw=1.2 if data is serie else 1.5, alpha=0.85)
    ax.set_ylabel(titulo, fontsize=8.5, color=color, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.yaxis.label.set_color(color)

axes[-1].set_xlabel("Dias desde el inicio")

# Marcar un año en el eje x
for ax in axes:
    for anio in [365, 730]:
        ax.axvline(anio, color=GRIS, lw=1, ls=":", alpha=0.5)
axes[0].text(182, serie.max() * 0.98, "Año 1", ha="center", fontsize=8, color=GRIS)
axes[0].text(547, serie.max() * 0.98, "Año 2", ha="center", fontsize=8, color=GRIS)

plt.tight_layout()
savefig("01-descomposicion-serie.png")


# ─── 02. Ventana deslizante: de serie a dataset supervisado ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=FONDO)
fig.suptitle(
    "Ventana Deslizante: Transformar una Serie en un Problema Supervisado",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

# Panel izquierdo: diagrama de la ventana
ax1 = axes[0]
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title("Como funciona la ventana", fontweight="bold", color=DARK, pad=8)

np.random.seed(7)
valores = np.round(100 + np.cumsum(np.random.normal(0, 2, 12)), 1)
n = len(valores)
W = 4  # tamaño de ventana
H = 1  # horizonte de prediccion

# Dibujar la serie como cajas
celda_w, celda_h, gap = 0.7, 0.25, 0.08
colores_celdas = [AZUL] * W + [VERDE] + [GRIS] * (n - W - 1)

for i, (v, c) in enumerate(zip(valores, colores_celdas)):
    x = 0.05 + i * (celda_w + gap) * 0.085
    rect = FancyBboxPatch(
        (x, 0.65),
        celda_w * 0.085,
        celda_h,
        boxstyle="round,pad=0.01",
        facecolor=c + "33",
        edgecolor=c,
        linewidth=1.5,
        transform=ax1.transAxes,
    )
    ax1.add_patch(rect)
    ax1.text(
        x + celda_w * 0.042,
        0.65 + celda_h / 2,
        str(v),
        ha="center",
        va="center",
        fontsize=7,
        color=c,
        fontweight="bold",
        transform=ax1.transAxes,
    )

ax1.text(
    0.05 + 2 * (celda_w + gap) * 0.085 * (W / 2),
    0.93,
    f"← Ventana (w={W}) →",
    ha="center",
    fontsize=9,
    color=AZUL,
    fontweight="bold",
    transform=ax1.transAxes,
)
ax1.text(
    0.05 + W * (celda_w + gap) * 0.085 + celda_w * 0.042,
    0.93,
    f"Target\n(h={H})",
    ha="center",
    fontsize=9,
    color=VERDE,
    fontweight="bold",
    transform=ax1.transAxes,
)

# Tres filas de ventana deslizante
window_data = []
for start in range(min(4, n - W - H + 1)):
    feat = valores[start : start + W]
    tgt = valores[start + W : start + W + H]
    window_data.append((feat, tgt))

y_row = 0.52
ax1.text(
    0.01,
    y_row + 0.06,
    "Dataset resultante (X → y):",
    fontsize=9,
    fontweight="bold",
    color=DARK,
    transform=ax1.transAxes,
)
for i, (feat, tgt) in enumerate(window_data):
    y_row -= 0.10
    feat_str = f"[{', '.join(str(f) for f in feat)}]"
    tgt_str = f"→  {tgt[0]}"
    ax1.text(
        0.04,
        y_row,
        f"X{i+1} = {feat_str}",
        fontsize=8.5,
        color=AZUL,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.60,
        y_row,
        tgt_str,
        fontsize=8.5,
        color=VERDE,
        fontweight="bold",
        transform=ax1.transAxes,
    )

# Panel derecho: el efecto del tamaño de ventana
ax2 = axes[1]
ax2.set_facecolor(FONDO)
n_datos = 5000
ventanas = [1, 3, 7, 14, 30, 60, 90, 180]
n_muestras = [n_datos - w - 1 for w in ventanas]
# Simular MAE segun el tamaño de ventana (parabolico simulado)
mae_sim = [18, 12, 8.5, 7.2, 6.8, 7.5, 8.9, 11.2]

color_bars = [VERDE if m == min(mae_sim) else AZUL for m in mae_sim]
bars = ax2.bar(
    [str(w) for w in ventanas], mae_sim, color=color_bars, edgecolor="white", alpha=0.85
)
ax2.set_xlabel("Tamaño de ventana (dias)")
ax2.set_ylabel("MAE en validacion")
ax2.set_title(
    f"Impacto del tamaño de ventana\nen el error de prediccion (n_total={n_datos})",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
for bar, v in zip(bars, mae_sim):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{v}",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=VERDE if v == min(mae_sim) else DARK,
    )
ax2.grid(axis="y", alpha=0.3)

# Agregar twin axis con numero de muestras
ax2b = ax2.twinx()
ax2b.plot(
    [str(w) for w in ventanas],
    n_muestras,
    "o--",
    color=NARANJA,
    lw=2,
    ms=6,
    label="Muestras disponibles",
)
ax2b.set_ylabel("Muestras en dataset", color=NARANJA)
ax2b.legend(fontsize=8, loc="upper right")

plt.tight_layout()
savefig("02-ventana-deslizante.png")


# ─── 03. Validación temporal correcta vs incorrecta ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=FONDO)
fig.suptitle(
    "Validacion Temporal: Por que el Split Aleatorio Causa Leakage Temporal",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

n_puntos = 100

for ax, (titulo, color_tren, color_val, tipo) in zip(
    axes,
    [
        (
            "Split Aleatorio\n(INCORRECTO — mezcla futuro con pasado)",
            ROJO,
            "#FCA5A5",
            "aleatorio",
        ),
        (
            "TimeSeriesSplit\n(CORRECTO — val siempre despues de train)",
            AZUL,
            "#BFDBFE",
            "temporal",
        ),
    ],
):
    ax.set_facecolor(FONDO)
    ax.set_title(
        titulo,
        fontsize=9,
        fontweight="bold",
        color=ROJO if tipo == "aleatorio" else AZUL,
        pad=6,
    )
    ax.set_xlim(-1, n_puntos + 1)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel("Indice temporal (tiempo →)")
    ax.set_yticks([])

    if tipo == "aleatorio":
        # 5 folds con indices aleatorios
        indices = np.arange(n_puntos)
        for fold in range(5):
            np.random.shuffle(indices)
            train_idx = indices[:70]
            val_idx = indices[70:]
            y = 5 - fold
            ax.scatter(
                train_idx,
                [y] * len(train_idx),
                c=color_tren,
                s=18,
                alpha=0.7,
                marker="|",
            )
            ax.scatter(
                val_idx, [y] * len(val_idx), c=color_val, s=18, alpha=0.7, marker="|"
            )
            ax.text(
                -0.5, y, f"F{fold+1}", ha="right", fontsize=8, color=DARK, va="center"
            )
        # Anotacion de problema
        ax.text(
            50,
            -0.2,
            "Datos futuros se filtran al train → score inflado artificialmente",
            ha="center",
            fontsize=8,
            color=ROJO,
            style="italic",
        )
    else:
        # TimeSeriesSplit: train crece, val siempre adelante
        fold_sizes = [20, 20, 20, 20, 20]
        train_end = 40
        for fold in range(5):
            val_start = train_end
            val_end = val_start + fold_sizes[fold]
            if val_end > n_puntos:
                break
            y = 5 - fold
            ax.barh(y, train_end, left=0, height=0.5, color=color_tren, alpha=0.7)
            ax.barh(
                y,
                val_end - val_start,
                left=val_start,
                height=0.5,
                color=color_val,
                alpha=0.9,
            )
            ax.axvline(
                val_start, ymin=(y - 0.25) / 6, ymax=(y + 0.25) / 6, color=VERDE, lw=2
            )
            ax.text(
                -0.5, y, f"F{fold+1}", ha="right", fontsize=8, color=DARK, va="center"
            )
            train_end = val_end

        # Leyenda
        p1 = mpatches.Patch(color=color_tren, alpha=0.7, label="Train")
        p2 = mpatches.Patch(color=color_val, alpha=0.9, label="Validacion")
        ax.legend(handles=[p1, p2], fontsize=8, loc="lower right")
        ax.text(
            50,
            -0.2,
            "El tiempo fluye de izquierda a derecha: val SIEMPRE despues de train",
            ha="center",
            fontsize=8,
            color=VERDE,
            style="italic",
        )

plt.tight_layout()
savefig("03-validacion-temporal.png")


# ─── 04. ACF y PACF: identificar lags relevantes ─────────────────────────────
np.random.seed(123)
n = 200

# Generar AR(2) process: y_t = 0.6*y_{t-1} + 0.3*y_{t-2} + eps
y = np.zeros(n)
eps = np.random.normal(0, 1, n)
for i in range(2, n):
    y[i] = 0.6 * y[i - 1] + 0.3 * y[i - 2] + eps[i]

# Calcular ACF y PACF manualmente
max_lags = 25


def acf_manual(x, max_lag):
    x = x - x.mean()
    c0 = np.sum(x**2) / len(x)
    acf_vals = []
    for lag in range(max_lag + 1):
        c = np.sum(x[: len(x) - lag] * x[lag:]) / len(x)
        acf_vals.append(c / c0)
    return np.array(acf_vals)


def pacf_yw(x, max_lag):
    """PACF por Yule-Walker (simplificado)."""
    acf_v = acf_manual(x, max_lag)
    pacf_vals = [1.0]
    for k in range(1, max_lag + 1):
        # Toeplitz matrix de ACF
        R = np.array([[acf_v[abs(i - j)] for j in range(k)] for i in range(k)])
        r = acf_v[1 : k + 1]
        try:
            phi = np.linalg.solve(R, r)
            pacf_vals.append(phi[-1])
        except np.linalg.LinAlgError:
            pacf_vals.append(0.0)
    return np.array(pacf_vals)


lags = np.arange(0, max_lags + 1)
acf_vals = acf_manual(y, max_lags)
pacf_vals = pacf_yw(y, max_lags)
conf_int = 1.96 / np.sqrt(n)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=FONDO)
fig.suptitle(
    "ACF y PACF: Detectar la Estructura de Dependencia Temporal",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

for ax, (vals, titulo, color, interpretacion) in zip(
    axes,
    [
        (
            acf_vals,
            "Autocorrelacion (ACF)",
            AZUL,
            "ACF decae exponencialmente → proceso AR\nLags significativos indican dependencias",
        ),
        (
            pacf_vals,
            "Autocorrelacion Parcial (PACF)",
            MORADO,
            "PACF corta en lag=2 → AR(2) (orden del proceso)\nUsado para elegir parametros de ARIMA",
        ),
    ],
):
    ax.set_facecolor(FONDO)
    # Barras
    ax.bar(lags[1:], vals[1:], color=color, alpha=0.75, edgecolor="white")
    # Bandas de confianza
    ax.axhline(conf_int, color=ROJO, lw=1.5, ls="--", label=f"IC 95% (±{conf_int:.3f})")
    ax.axhline(-conf_int, color=ROJO, lw=1.5, ls="--")
    ax.axhline(0, color=DARK, lw=0.8)
    # Marcar lags significativos
    significativos = np.where(np.abs(vals[1:]) > conf_int)[0] + 1
    for lag in significativos[:5]:
        ax.text(
            lag,
            vals[lag] + 0.02 * np.sign(vals[lag]),
            str(lag),
            ha="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlacion")
    ax.set_title(titulo, fontsize=10, fontweight="bold", color=color)
    ax.set_ylim(-0.6, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.5,
        0.05,
        interpretacion,
        ha="center",
        fontsize=8,
        color=GRIS,
        style="italic",
        transform=ax.transAxes,
    )

plt.tight_layout()
savefig("04-acf-pacf.png")


# ─── 05. Comparación de modelos de forecasting ───────────────────────────────
np.random.seed(88)
n_train = 150
n_test = 30
t_all = np.arange(n_train + n_test)

# Serie real con tendencia y estacionalidad
true_series = (
    50
    + 0.2 * t_all
    + 8 * np.sin(2 * np.pi * t_all / 52)
    + np.random.normal(0, 2.5, len(t_all))
)

y_train = true_series[:n_train]
y_test = true_series[n_train:]
t_train = t_all[:n_train]
t_test = t_all[n_train:]

# Naive seasonal (lag=52)
lag = 52
naive_pred = np.array([y_train[n_train - lag + (i % lag)] for i in range(n_test)])

# Tendencia lineal
coef = np.polyfit(t_train, y_train, 1)
lineal_pred = np.polyval(coef, t_test)

# LGBM simulado (mejor)
lgbm_pred = true_series[n_train:] + np.random.normal(0, 1.8, n_test)

# LSTM simulado (bueno, similar a LGBM)
lstm_pred = (
    true_series[n_train:]
    + np.random.normal(0, 2.2, n_test)
    + 0.5 * np.sin(2 * np.pi * np.arange(n_test) / 7)
)


def mae(a, b):
    return np.mean(np.abs(a - b))


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


fig, axes = plt.subplots(
    2, 1, figsize=(13, 9), facecolor=FONDO, gridspec_kw={"height_ratios": [2, 1]}
)
fig.suptitle(
    "Comparacion de Modelos de Forecasting: Naive, Lineal, LGBM, LSTM",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

ax1 = axes[0]
ax1.set_facecolor(FONDO)
ax1.plot(t_train, y_train, color=DARK, lw=1.5, alpha=0.7, label="Serie historica")
ax1.plot(t_test, y_test, color=DARK, lw=2.5, alpha=0.9, label="Valores reales")
ax1.plot(
    t_test,
    naive_pred,
    color=GRIS,
    lw=2,
    ls=":",
    label=f"Naive (MAE={mae(y_test, naive_pred):.2f})",
)
ax1.plot(
    t_test,
    lineal_pred,
    color=AMARILLO,
    lw=2,
    ls="--",
    label=f"Lineal (MAE={mae(y_test, lineal_pred):.2f})",
)
ax1.plot(
    t_test,
    lgbm_pred,
    color=VERDE,
    lw=2.5,
    label=f"LGBM+lags (MAE={mae(y_test, lgbm_pred):.2f})",
)
ax1.plot(
    t_test,
    lstm_pred,
    color=MORADO,
    lw=2.5,
    ls="-.",
    label=f"LSTM (MAE={mae(y_test, lstm_pred):.2f})",
)
ax1.axvline(n_train, color=ROJO, lw=2.5, ls="--", label="Inicio de test")
ax1.fill_betweenx(
    [y_train.min() - 5, y_train.max() + 5],
    n_train,
    n_train + n_test,
    alpha=0.07,
    color=ROJO,
)
ax1.set_ylabel("Valor")
ax1.legend(fontsize=8.5, ncol=2)
ax1.set_title(
    "Prediccion en el horizonte de test (30 pasos)",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax1.grid(True, alpha=0.25)

# Panel de barras de MAE por modelo
ax2 = axes[1]
ax2.set_facecolor(FONDO)
modelos = ["Naive\nSeasonal", "Lineal", "LGBM\n+lags", "LSTM"]
maes = [
    mae(y_test, naive_pred),
    mae(y_test, lineal_pred),
    mae(y_test, lgbm_pred),
    mae(y_test, lstm_pred),
]
rmses = [
    rmse(y_test, naive_pred),
    rmse(y_test, lineal_pred),
    rmse(y_test, lgbm_pred),
    rmse(y_test, lstm_pred),
]
colors_b = [GRIS, AMARILLO, VERDE, MORADO]
x = np.arange(len(modelos))
ax2.bar(x - 0.2, maes, 0.35, color=colors_b, alpha=0.85, edgecolor="white", label="MAE")
ax2.bar(
    x + 0.2, rmses, 0.35, color=colors_b, alpha=0.55, edgecolor="white", label="RMSE"
)
ax2.set_xticks(x)
ax2.set_xticklabels(modelos)
ax2.set_ylabel("Error")
ax2.set_title("MAE y RMSE por modelo", fontsize=9, fontweight="bold", color=DARK)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
for xi, (m, r) in enumerate(zip(maes, rmses)):
    ax2.text(xi - 0.2, m + 0.05, f"{m:.2f}", ha="center", fontsize=8, color=DARK)
    ax2.text(xi + 0.2, r + 0.05, f"{r:.2f}", ha="center", fontsize=8, color=GRIS)

plt.tight_layout()
savefig("05-comparacion-modelos-forecast.png")


# ─── 06. Features de lag y rolling: ingeniería para series temporales ─────────
np.random.seed(55)
n_feat = 120
fecha_base = np.arange(n_feat)
serie_feat = (
    100
    + 0.3 * fecha_base
    + 10 * np.sin(2 * np.pi * fecha_base / 30)
    + np.random.normal(0, 3, n_feat)
)

# Crear features
lag1 = np.roll(serie_feat, 1)
lag1[0] = np.nan
lag7 = np.roll(serie_feat, 7)
lag7[:7] = np.nan
lag30 = np.roll(serie_feat, 30)
lag30[:30] = np.nan
roll7 = np.array(
    [
        np.mean(serie_feat[max(0, i - 7) : i]) if i >= 7 else np.nan
        for i in range(n_feat)
    ]
)
roll30 = np.array(
    [
        np.mean(serie_feat[max(0, i - 30) : i]) if i >= 30 else np.nan
        for i in range(n_feat)
    ]
)
diff1 = np.concatenate([[np.nan], np.diff(serie_feat)])
diff7 = np.concatenate([[np.nan] * 7, serie_feat[7:] - serie_feat[:-7]])

fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=FONDO)
fig.suptitle(
    "Features de Lag y Rolling: La Base del Feature Engineering Temporal",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

# Panel 1: Serie + lags
ax1 = axes[0, 0]
ax1.set_facecolor(FONDO)
ax1.plot(fecha_base, serie_feat, color=AZUL, lw=2, label="y_t (original)")
ax1.plot(fecha_base, lag1, color=VERDE, lw=1.5, ls="--", label="lag_1 (y_{t-1})")
ax1.plot(fecha_base, lag7, color=NARANJA, lw=1.5, ls="-.", label="lag_7 (y_{t-7})")
ax1.plot(fecha_base, lag30, color=MORADO, lw=1.5, ls=":", label="lag_30 (y_{t-30})")
ax1.set_title(
    "Lags: valores pasados como features", fontsize=9, fontweight="bold", color=DARK
)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)
ax1.set_xlabel("Tiempo")
ax1.set_ylabel("Valor")

# Panel 2: Serie + rolling means
ax2 = axes[0, 1]
ax2.set_facecolor(FONDO)
ax2.plot(fecha_base, serie_feat, color=AZUL, lw=1.5, alpha=0.5, label="y_t (original)")
ax2.plot(fecha_base, roll7, color=VERDE, lw=2.5, label="rolling_mean_7")
ax2.plot(fecha_base, roll30, color=NARANJA, lw=2.5, ls="--", label="rolling_mean_30")
ax2.set_title(
    "Rolling Mean: suavizado de tendencia local",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Valor")

# Panel 3: Diferenciacion
ax3 = axes[1, 0]
ax3.set_facecolor(FONDO)
ax3.plot(fecha_base, diff1, color=CIAN, lw=1.5, label="diff_1 (y_t - y_{t-1})")
ax3.plot(fecha_base, diff7, color=ROSA, lw=1.5, ls="--", label="diff_7 (y_t - y_{t-7})")
ax3.axhline(0, color=GRIS, lw=1, ls=":")
ax3.set_title(
    "Diferenciacion: hacer la serie estacionaria",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25)
ax3.set_xlabel("Tiempo")
ax3.set_ylabel("Diferencia")

# Panel 4: Tabla de features resultante
ax4 = axes[1, 1]
ax4.set_facecolor(FONDO)
ax4.axis("off")
ax4.set_title(
    "Catalogo de features temporales", fontsize=9, fontweight="bold", color=DARK, pad=8
)

features_tabla = [
    ("Categoria", "Feature", "Formula", "Uso"),
    ("Lag", "lag_1..N", "y_{t-k}", "Dependencia directa"),
    ("Rolling", "roll_mean_k", "mean(y[t-k:t])", "Tendencia local"),
    ("Rolling", "roll_std_k", "std(y[t-k:t])", "Volatilidad"),
    ("Rolling", "roll_min/max_k", "min/max(y[t-k:t])", "Extremos locales"),
    ("Diferencia", "diff_1", "y_t - y_{t-1}", "Cambio instantaneo"),
    ("Diferencia", "diff_k", "y_t - y_{t-k}", "Cambio estacional"),
    ("Fecha", "dia_semana", "fecha.dayofweek", "Efecto semanal"),
    ("Fecha", "mes", "fecha.month", "Efecto mensual"),
    ("Fecha", "es_feriado", "0 / 1", "Dias especiales"),
    ("Expansion", "cumsum", "sum(y[0:t])", "Acumulado"),
]

y = 0.97
for i, (cat, feat, formula, uso) in enumerate(features_tabla):
    bg = AZUL + "22" if i == 0 else ("#EFF6FF" if i % 2 == 0 else FONDO)
    rect = FancyBboxPatch(
        (0, y - 0.075),
        1.0,
        0.072,
        boxstyle="round,pad=0.003",
        facecolor=bg,
        edgecolor="none",
        transform=ax4.transAxes,
    )
    ax4.add_patch(rect)
    bold = i == 0
    for text, xpos, col in [
        (cat, 0.01, AZUL if i > 0 else DARK),
        (feat, 0.20, DARK),
        (formula, 0.48, MORADO),
        (uso, 0.67, GRIS),
    ]:
        ax4.text(
            xpos,
            y - 0.032,
            text,
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="bold" if bold else "normal",
            color=col,
            transform=ax4.transAxes,
        )
    y -= 0.082

plt.tight_layout()
savefig("06-features-lag-rolling.png")


# ─── 07. Arquitectura RNN y LSTM ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 7), facecolor=FONDO)
fig.suptitle(
    "RNN vs LSTM: Memoria a Corto y Largo Plazo",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

# Panel izquierdo: RNN desplegada en el tiempo
ax1 = axes[0]
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title("RNN desplegada en el tiempo", fontweight="bold", color=DARK, pad=8)

t_steps = 4
cell_w, cell_h = 0.12, 0.15
gap_x = 0.20

for i in range(t_steps):
    x = 0.12 + i * gap_x
    # Celda RNN
    rect = FancyBboxPatch(
        (x, 0.45),
        cell_w,
        cell_h,
        boxstyle="round,pad=0.02",
        facecolor="#BFDBFE",
        edgecolor=AZUL,
        linewidth=2,
        transform=ax1.transAxes,
    )
    ax1.add_patch(rect)
    ax1.text(
        x + cell_w / 2,
        0.45 + cell_h / 2,
        f"h_{i}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=AZUL,
        transform=ax1.transAxes,
    )

    # Input x_t
    ax1.annotate(
        "",
        xy=(x + cell_w / 2, 0.45),
        xytext=(x + cell_w / 2, 0.28),
        arrowprops=dict(arrowstyle="->", color=VERDE, lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    ax1.text(
        x + cell_w / 2,
        0.22,
        f"x_{i}",
        ha="center",
        fontsize=10,
        color=VERDE,
        fontweight="bold",
        transform=ax1.transAxes,
    )

    # Output y_t
    ax1.annotate(
        "",
        xy=(x + cell_w / 2, 0.72),
        xytext=(x + cell_w / 2, 0.60),
        arrowprops=dict(arrowstyle="->", color=NARANJA, lw=2),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    ax1.text(
        x + cell_w / 2,
        0.75,
        f"ŷ_{i}",
        ha="center",
        fontsize=10,
        color=NARANJA,
        fontweight="bold",
        transform=ax1.transAxes,
    )

    # Conexion h_{t-1} -> h_t
    if i > 0:
        ax1.annotate(
            "",
            xy=(x + 0.01, 0.525),
            xytext=(x - gap_x + cell_w - 0.01, 0.525),
            arrowprops=dict(arrowstyle="->", color=MORADO, lw=2.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
        )

ax1.text(
    0.10,
    0.525,
    "h_{t-1}",
    ha="center",
    fontsize=9,
    color=MORADO,
    transform=ax1.transAxes,
)
ax1.text(
    0.5,
    0.88,
    "Problema: gradiente desaparece en\nsecuencias largas (> 50-100 pasos)",
    ha="center",
    fontsize=9,
    color=ROJO,
    style="italic",
    transform=ax1.transAxes,
)
ax1.text(
    0.5,
    0.10,
    "h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)",
    ha="center",
    fontsize=9,
    color=AZUL,
    fontweight="bold",
    transform=ax1.transAxes,
)

# Panel derecho: LSTM compuertas
ax2 = axes[1]
ax2.set_facecolor(FONDO)
ax2.axis("off")
ax2.set_title(
    "LSTM: celda con 3 compuertas (Gates)", fontweight="bold", color=DARK, pad=8
)

# Celda LSTM principal
rect_lstm = FancyBboxPatch(
    (0.10, 0.20),
    0.80,
    0.55,
    boxstyle="round,pad=0.03",
    facecolor="#EDE9FE",
    edgecolor=MORADO,
    linewidth=2.5,
    transform=ax2.transAxes,
)
ax2.add_patch(rect_lstm)

# Las 3 compuertas
gates = [
    (
        0.22,
        0.60,
        "Forget\nGate (f_t)",
        ROJO,
        "Que olvidar\ndel estado anterior\nf_t = σ(W_f·[h,x]+b)",
    ),
    (
        0.50,
        0.60,
        "Input\nGate (i_t)",
        VERDE,
        "Que informacion\nnueva almacenar\ni_t = σ(W_i·[h,x]+b)",
    ),
    (
        0.78,
        0.60,
        "Output\nGate (o_t)",
        AZUL,
        "Que parte del\nestado de celda\nenviar a h_t",
    ),
]
for xg, yg, nombre, color, desc in gates:
    circ = plt.Circle(
        (xg, yg),
        0.065,
        facecolor=color + "44",
        edgecolor=color,
        linewidth=2,
        transform=ax2.transAxes,
    )
    ax2.add_patch(circ)
    ax2.text(
        xg,
        yg,
        nombre.split("\n")[1],
        ha="center",
        va="center",
        fontsize=8,
        color=color,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        xg,
        yg - 0.16,
        desc,
        ha="center",
        fontsize=7,
        color=DARK,
        multialignment="center",
        transform=ax2.transAxes,
    )

# Cell state (linea horizontal)
ax2.annotate(
    "",
    xy=(0.90, 0.72),
    xytext=(0.10, 0.72),
    arrowprops=dict(arrowstyle="->", color=NARANJA, lw=3.5),
    xycoords="axes fraction",
    textcoords="axes fraction",
)
ax2.text(
    0.50,
    0.76,
    "Cell state C_t   (memoria a largo plazo)",
    ha="center",
    fontsize=9,
    color=NARANJA,
    fontweight="bold",
    transform=ax2.transAxes,
)

# Entradas y salidas
ax2.text(
    0.50,
    0.15,
    "[h_{t-1}, x_t]",
    ha="center",
    fontsize=10,
    color=DARK,
    fontweight="bold",
    transform=ax2.transAxes,
)
ax2.annotate(
    "",
    xy=(0.50, 0.20),
    xytext=(0.50, 0.16),
    arrowprops=dict(arrowstyle="->", color=DARK, lw=2),
    xycoords="axes fraction",
    textcoords="axes fraction",
)
ax2.text(
    0.50,
    0.82,
    "h_t  (memoria a corto plazo / salida)",
    ha="center",
    fontsize=9,
    color=MORADO,
    fontweight="bold",
    transform=ax2.transAxes,
)

ax2.text(
    0.50,
    0.08,
    "La compuerta Forget decide que olvidar → resuelve el vanishing gradient",
    ha="center",
    fontsize=8.5,
    color=ROJO,
    style="italic",
    transform=ax2.transAxes,
)

plt.tight_layout()
savefig("07-rnn-lstm-arquitectura.png")


# ─── 08. Error por horizonte de predicción ───────────────────────────────────
np.random.seed(33)
horizontes = np.arange(1, 31)

# MAE aumenta con el horizonte (diferentes tasas por modelo)
mae_naive = 2.5 + 0.30 * horizontes + np.random.normal(0, 0.3, len(horizontes))
mae_lgbm = 1.8 + 0.12 * horizontes + np.random.normal(0, 0.2, len(horizontes))
mae_lstm = 1.9 + 0.14 * horizontes + np.random.normal(0, 0.25, len(horizontes))
mae_direct = 1.7 + 0.09 * horizontes + np.random.normal(0, 0.2, len(horizontes))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=FONDO)
fig.suptitle(
    "Error por Horizonte: El Error Aumenta con la Distancia en el Tiempo",
    fontsize=13,
    fontweight="bold",
    color=DARK,
)

ax1 = axes[0]
ax1.set_facecolor(FONDO)
ax1.plot(horizontes, mae_naive, "o-", color=GRIS, lw=2, ms=5, label="Naive Seasonal")
ax1.plot(horizontes, mae_lgbm, "s-", color=VERDE, lw=2, ms=5, label="LGBM (recursivo)")
ax1.plot(horizontes, mae_lstm, "^-", color=MORADO, lw=2, ms=5, label="LSTM (seq2seq)")
ax1.plot(horizontes, mae_direct, "D-", color=AZUL, lw=2.5, ms=5, label="LGBM (directo)")
ax1.fill_between(
    horizontes,
    mae_lgbm,
    mae_direct,
    alpha=0.1,
    color=AZUL,
    label="Ganancia estrategia directa",
)
ax1.set_xlabel("Horizonte (pasos adelante)")
ax1.set_ylabel("MAE")
ax1.set_title(
    "MAE vs Horizonte\npor modelo y estrategia",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax1.legend(fontsize=8.5)
ax1.grid(True, alpha=0.3)

# Panel derecho: recursivo vs directo vs DIRMO
ax2 = axes[1]
ax2.set_facecolor(FONDO)
ax2.axis("off")
ax2.set_title(
    "Estrategias de multi-step forecasting", fontweight="bold", color=DARK, pad=8
)

estrategias = [
    (
        "Recursiva",
        NARANJA,
        "Entrena un modelo para h=1.\nUsa la prediccion t+1 como input\npara predecir t+2, etc.\nError se acumula en cada paso.",
    ),
    (
        "Directa",
        VERDE,
        "Entrena un modelo separado\npara cada horizonte h.\nMejor para horizontes lejanos.\nMas costoso (N modelos).",
    ),
    (
        "DIRMO\n(Multi-Output)",
        AZUL,
        "Un solo modelo predice\ntodos los horizontes a la vez.\nEj: LGBM MultiOutputRegressor\no LSTM seq2seq.",
    ),
    (
        "Hibrida",
        MORADO,
        "Combina recursiva para h corto\ny directa para h largo.\nBalance entre costo y precision.",
    ),
]
y = 0.97
for nombre, color, desc in estrategias:
    rect = FancyBboxPatch(
        (0.01, y - 0.22),
        0.98,
        0.21,
        boxstyle="round,pad=0.02",
        facecolor=color + "18",
        edgecolor=color,
        linewidth=1.8,
        transform=ax2.transAxes,
    )
    ax2.add_patch(rect)
    ax2.text(
        0.04,
        y - 0.03,
        nombre,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=color,
        transform=ax2.transAxes,
    )
    ax2.text(
        0.04,
        y - 0.10,
        desc,
        ha="left",
        va="top",
        fontsize=7.5,
        color=DARK,
        transform=ax2.transAxes,
    )
    y -= 0.25

plt.tight_layout()
savefig("08-error-horizonte.png")


# ─── 09. Dashboard resumen ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=FONDO)
fig.suptitle(
    "Tema 18: Series Temporales y Datos Secuenciales — Dashboard",
    fontsize=15,
    fontweight="bold",
    color=DARK,
    y=0.97,
)

gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.50,
    wspace=0.40,
    left=0.06,
    right=0.97,
    top=0.91,
    bottom=0.06,
)

# Panel 1: flujo de trabajo en series temporales
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(FONDO)
ax1.axis("off")
ax1.set_title("Flujo de trabajo en ST", fontsize=9, fontweight="bold", color=DARK)
pasos_flujo = [
    (
        "1. EDA temporal",
        AZUL,
        "Visualizar, estacionalidad,\nACF/PACF, test de estacionariedad",
    ),
    ("2. Baseline naive", VERDE, "Naive, media, ultimo valor,\nSeasonal naive"),
    ("3. Features", NARANJA, "Lags, rolling stats,\nfeatures de fecha"),
    ("4. Modelo ML", MORADO, "LGBM/XGB con lags,\nvalidacion temporal"),
    ("5. Modelo DL", CIAN, "LSTM/GRU seq2seq\npara patrones complejos"),
    ("6. Ensemble", ROSA, "Blend ML + DL\n+ modelo estadistico"),
]
y = 0.96
for nombre, color, desc in pasos_flujo:
    rect = FancyBboxPatch(
        (0, y - 0.14),
        1.0,
        0.13,
        boxstyle="round,pad=0.01",
        facecolor=color + "22",
        edgecolor=color,
        linewidth=1.2,
        transform=ax1.transAxes,
    )
    ax1.add_patch(rect)
    ax1.text(
        0.04,
        y - 0.035,
        nombre,
        fontsize=8,
        fontweight="bold",
        color=color,
        transform=ax1.transAxes,
    )
    ax1.text(0.04, y - 0.095, desc, fontsize=6.5, color=DARK, transform=ax1.transAxes)
    y -= 0.16

# Panel 2: metricas de forecasting
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(FONDO)
ax2.axis("off")
ax2.set_title("Metricas de Forecasting", fontsize=9, fontweight="bold", color=DARK)
metricas_tabla = [
    ("MAE", "mean|y - ŷ|", "Facil de interpretar,\nmisma unidad que y"),
    ("RMSE", "sqrt(mean(y-ŷ)²)", "Penaliza errores grandes,\nsensible a outliers"),
    ("MAPE", "mean|y-ŷ|/y * 100", "Porcentual, util si y\nvaria mucho en magnitud"),
    ("sMAPE", "2|y-ŷ|/(|y|+|ŷ|)", "MAPE simetrico,\nevita division por 0"),
    ("MASE", "MAE / MAE_naive", "Normalizado por naive,\n<1 = mejor que naive"),
    ("WQL", "Weighted Quantile", "Para intervalos de\nconfianza (prob forecast)"),
]
y = 0.97
for nombre, formula, nota in metricas_tabla:
    ax2.text(
        0.01,
        y,
        nombre,
        fontsize=9,
        fontweight="bold",
        color=AZUL,
        transform=ax2.transAxes,
    )
    ax2.text(0.22, y, formula, fontsize=8, color=DARK, transform=ax2.transAxes)
    ax2.text(0.01, y - 0.06, nota, fontsize=7, color=GRIS, transform=ax2.transAxes)
    y -= 0.155

# Panel 3: mini series comparando estacionaria vs no estacionaria
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(FONDO)
np.random.seed(21)
t3 = np.arange(80)
estacionaria = np.random.normal(0, 1, 80)
no_estacionaria = np.cumsum(np.random.normal(0, 1, 80)) + 0.05 * t3
ax3.plot(
    t3, estacionaria + 5, color=VERDE, lw=1.5, label="Estacionaria (media constante)"
)
ax3.plot(t3, no_estacionaria, color=ROJO, lw=1.5, label="No estacionaria (drift)")
ax3.axhline(np.mean(estacionaria + 5), color=VERDE, lw=1, ls="--", alpha=0.5)
ax3.set_title(
    "Estacionariedad: requisito\nde ARIMA y muchos modelos",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
ax3.legend(fontsize=7.5)
ax3.grid(True, alpha=0.25)
ax3.text(
    40,
    -8,
    "Diferenciacion (diff) convierte\nno estacionaria -> estacionaria",
    ha="center",
    fontsize=8,
    color=NARANJA,
    style="italic",
)

# Panel 4: librerías y herramientas
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor(FONDO)
ax4.axis("off")
ax4.set_title(
    "Librerias y herramientas para Series Temporales",
    fontsize=9,
    fontweight="bold",
    color=DARK,
)
libs = [
    (
        "statsmodels",
        AZUL,
        "ARIMA, SARIMA, descomposicion STL, tests ADF/KPSS, diagnosticos",
    ),
    (
        "Prophet",
        VERDE,
        "Facebook Prophet: tendencia + estacionalidad + feriados, muy practico",
    ),
    (
        "LightGBM + lags",
        NARANJA,
        "Transformar ST en supervisado con lags; gana la mayoria de competencias",
    ),
    (
        "PyTorch LSTM",
        MORADO,
        "RNN/LSTM/GRU desde cero; control total del modelo secuencial",
    ),
    (
        "Darts",
        CIAN,
        "API unificada para ARIMA, Prophet, LGBM, LSTM, Transformers en ST",
    ),
    (
        "sktime",
        ROSA,
        "scikit-learn compatible para forecasting, clasificacion y regresion en ST",
    ),
    (
        "GluonTS",
        AMARILLO,
        "Amazon: modelos probabilisticos de ST (DeepAR, Transformer)",
    ),
]
y = 0.96
for nombre, color, desc in libs:
    rect = FancyBboxPatch(
        (0.00, y - 0.12),
        1.00,
        0.115,
        boxstyle="round,pad=0.005",
        facecolor=color + "15",
        edgecolor=color,
        linewidth=1.2,
        transform=ax4.transAxes,
    )
    ax4.add_patch(rect)
    ax4.text(
        0.01,
        y - 0.04,
        nombre,
        fontsize=9,
        fontweight="bold",
        color=color,
        transform=ax4.transAxes,
    )
    ax4.text(0.18, y - 0.04, desc, fontsize=8, color=DARK, transform=ax4.transAxes)
    y -= 0.135

# Panel 5: Errores comunes
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(FONDO)
ax5.axis("off")
ax5.set_title(
    "Errores comunes\nen competencias ST", fontsize=9, fontweight="bold", color=DARK
)
errores = [
    ("Random split\nen vez de temporal", ROJO),
    ("Lags sin cuidado de\n'look-ahead' leakage", ROJO),
    ("Ignorar estacionalidad\nen el baseline", NARANJA),
    ("No diferenciar target\ncon drift fuerte", NARANJA),
    ("LSTM sin suficientes\ndatos (> 10k pasos)", NARANJA),
    ("No escalar features\npara LSTM/GRU", AMARILLO),
    ("Evaluar con MAPE\ncuando y ≈ 0", AMARILLO),
]
y = 0.96
for error, color in errores:
    ax5.text(0.05, y, "▸", fontsize=10, color=color, transform=ax5.transAxes)
    ax5.text(0.12, y, error, fontsize=8, color=DARK, transform=ax5.transAxes)
    y -= 0.135

savefig("09-dashboard.png")

print("\nTodos los graficos del tema 18 generados en:", OUT)
