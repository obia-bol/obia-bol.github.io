---
title: "5. Introduccion a Machine Learning"
description: "Conceptos clave de aprendizaje supervisado/no supervisado, overfitting, sesgo-varianza y preparacion de caracteristicas."
pubDate: "May 05 2026"
badge: "Fase 2"
tags: ["Machine Learning", "Fundamentos", "Modelado"]
---

## Teoria

Machine Learning permite que un modelo aprenda patrones desde datos.

Pilares iniciales:

- Supervisado vs no supervisado.
- Split de entrenamiento, validacion y prueba.
- Overfitting vs underfitting.
- Bias vs variance.
- Feature engineering.
- Normalizacion y estandarizacion.

## Guia practica extensa de Machine Learning

Esta seccion esta pensada para que pases de teoria a implementacion real en competencias.

### 1) Supervisado vs no supervisado (con ejemplos)

Aprendizaje supervisado:

- Tienes una variable objetivo (target).
- Quieres predecir una etiqueta o valor.
- Ejemplos: clasificacion de spam, prediccion de precio.

Aprendizaje no supervisado:

- No tienes target.
- Buscas estructura oculta en datos.
- Ejemplos: clustering de clientes, deteccion de patrones.

Ejemplo supervisado (clasificacion):

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=12, n_informative=6, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
```

Ejemplo no supervisado (clustering):

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.2, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print("Primeras etiquetas:", labels[:10])
```

### 2) Split correcto: train / validation / test

Buena practica:

- `train`: entrenar.
- `validation`: comparar modelos y ajustar decisiones.
- `test`: evaluar al final, una sola vez.

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Paso 1: separar test final
X_temp, X_test, y_temp, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

# Paso 2: separar train y validation
X_train, X_val, y_train, y_val = train_test_split(
	X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(X_train.shape, X_val.shape, X_test.shape)  # 60/20/20
```

### 3) Overfitting vs underfitting

Señales tipicas:

- Overfitting: train muy alto, validacion baja.
- Underfitting: train y validacion bajos.

Mini experimento con complejidad de modelo:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Datos sinteticos
rng = np.random.RandomState(42)
X = np.sort(6 * rng.rand(200, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * rng.randn(200)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

for degree in [1, 3, 12]:
	model = Pipeline([
		("poly", PolynomialFeatures(degree=degree)),
		("lin", LinearRegression())
	])
	model.fit(X_train, y_train)
	train_mse = mean_squared_error(y_train, model.predict(X_train))
	val_mse = mean_squared_error(y_val, model.predict(X_val))
	print(f"degree={degree} | train_mse={train_mse:.4f} | val_mse={val_mse:.4f}")
```

Interpretacion esperada:

- `degree=1`: puede subajustar.
- `degree=12`: puede sobreajustar.
- `degree=3`: suele quedar mas equilibrado.

### 4) Bias vs variance en lenguaje practico

- Alto bias: modelo demasiado simple, no captura patrones.
- Alta variance: modelo demasiado sensible al ruido.

Acciones para reducir bias:

- aumentar capacidad del modelo,
- mejores features,
- reducir regularizacion excesiva.

Acciones para reducir variance:

- mas datos,
- regularizacion,
- simplificar modelo,
- validacion cruzada,
- ensembling.

### 5) Normalizacion y estandarizacion

Regla rapida:

- `StandardScaler`: centra en media 0 y desv estandar 1.
- `MinMaxScaler`: lleva valores a un rango (ejemplo 0 a 1).

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = np.array([
	[10, 1000],
	[12, 1200],
	[20, 3000],
], dtype=float)

std = StandardScaler().fit_transform(X)
mm = MinMaxScaler().fit_transform(X)

print("StandardScaler:\n", std)
print("MinMaxScaler:\n", mm)
```

Importante: el scaler se ajusta solo con train (`fit` en train, `transform` en val/test).

### 6) Feature engineering basico y util

```python
import pandas as pd

df = pd.DataFrame({
	"age": [18, 25, 40, 60],
	"income": [1000, 3000, 7000, 12000],
	"city": ["A", "B", "A", "C"]
})

# Features numericas
df["income_per_age"] = df["income"] / df["age"]
df["is_senior"] = (df["age"] >= 50).astype(int)

# One-hot encoding
df = pd.get_dummies(df, columns=["city"], drop_first=True)
print(df)
```

Buenas features suelen dar mejoras mas estables que tuning agresivo.

### 7) Metricas segun problema

Clasificacion:

- Accuracy
- Precision / Recall / F1
- ROC-AUC

Regresion:

- MAE
- RMSE
- R2

```python
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score,
	mean_absolute_error, root_mean_squared_error, r2_score
)

# Clasificacion
y_true_c = [0, 1, 1, 0, 1]
y_pred_c = [0, 1, 0, 0, 1]

print("Accuracy:", accuracy_score(y_true_c, y_pred_c))
print("Precision:", precision_score(y_true_c, y_pred_c))
print("Recall:", recall_score(y_true_c, y_pred_c))
print("F1:", f1_score(y_true_c, y_pred_c))

# Regresion
y_true_r = [10.0, 12.0, 15.0, 20.0]
y_pred_r = [9.5, 11.8, 14.0, 21.5]

print("MAE:", mean_absolute_error(y_true_r, y_pred_r))
print("RMSE:", root_mean_squared_error(y_true_r, y_pred_r))
print("R2:", r2_score(y_true_r, y_pred_r))
```

### 8) Baseline reproducible de principio a fin

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("train.csv")
target = "target"

X = df.drop(columns=[target])
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = X_train.select_dtypes(include=["number"]).columns
cat_cols = X_train.select_dtypes(exclude=["number"]).columns

num_pipe = Pipeline([
	("imputer", SimpleImputer(strategy="median")),
	("scaler", StandardScaler())
])

cat_pipe = Pipeline([
	("imputer", SimpleImputer(strategy="most_frequent")),
	("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
	("num", num_pipe, num_cols),
	("cat", cat_pipe, cat_cols)
])

model = Pipeline([
	("prep", preprocess),
	("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)
pred = model.predict(X_val)
print("F1 validation:", f1_score(y_val, pred))
```

Con este baseline ya puedes competir de forma ordenada.

## Por que importa en competencias de IA

En competencias, la diferencia entre una solucion promedio y una fuerte suele estar en validacion correcta y buenas features.

## Recursos recomendados

- [**Hands-On Machine Learning (Geron, O'Reilly)**](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/): el libro practico mas completo de ML con sklearn y Keras
- [**Kaggle Learn — Intro to Machine Learning**](https://www.kaggle.com/learn/intro-to-machine-learning): introduccion practica gratuita
- [**Kaggle Learn — Intermediate Machine Learning**](https://www.kaggle.com/learn/intermediate-machine-learning): validacion, pipelines y manejo de datos faltantes
- [**Machine Learning Specialization (Andrew Ng, Coursera)**](https://www.coursera.org/specializations/machine-learning-introduction): el curso de ML mas conocido del mundo

## Ejercicios practicos

- Probar diferentes ratios de train/val/test.
- Comparar desempeno con y sin escalado.
- Diagnosticar overfitting con curvas de aprendizaje.

## Mini-proyectos

- Baseline completo para dataset tabular: preprocessing + modelo + evaluacion.

### Como resolver el mini-proyecto (resumen practico)

Objetivo: construir una solucion base robusta y reproducible en menos de 2 horas.

Pasos sugeridos:

1. Definir problema y metrica oficial.
2. Separar train/validation correctamente.
3. Implementar preprocessing con `ColumnTransformer`.
4. Entrenar 1 baseline sencillo (por ejemplo RandomForest o XGBoost si aplica).
5. Evaluar en validation y guardar resultados.
6. Documentar 3 mejoras candidatas para la siguiente iteracion.

Estructura minima del entregable:

- `baseline.ipynb` o `baseline.py` ejecutable.
- Tabla de metricas (train vs validation).
- Lista de decisiones tecnicas tomadas.

Checklist final:

- El codigo corre de principio a fin sin edicion manual.
- No hay leakage evidente.
- La metrica reportada coincide con el objetivo de competencia.

## Errores comunes

- Evaluar sobre el mismo set usado para entrenar.
- Hacer feature engineering con informacion del test.
- Cambiar demasiadas cosas a la vez sin control experimental.

## Seccion avanzada (opcional)

Estudia tecnicas de validacion estratificada y validacion temporal para distintos tipos de datos.

## Ruta sugerida

1. Dominar split y metricas.
2. Implementar baseline reproducible.
3. Mejorar con features y tuning controlado.

## Desarrollo extendido para implementacion real

### Marco mental para ML competitivo

Cuando estudies este tema, piensa siempre en el pipeline completo, no en una tecnica aislada. Un buen resultado competitivo requiere:

- Datos confiables y limpios.
- Validacion sin fuga de informacion.
- Baseline fuerte y bien documentado.
- Iteracion controlada de mejoras.
- Seleccion final por evidencia, no por intuicion.

### Flujo recomendado de trabajo

1. Comprende el problema y la metrica oficial.
2. Construye un baseline en menos de 1 hora.
3. Analiza errores por segmentos de datos.
4. Propone 3 mejoras concretas (features, modelo, validacion).
5. Mide cada cambio por separado.
6. Conserva solo lo que mejora de forma consistente.

### Decision tecnica: como elegir entre alternativas

Preguntas practicas que debes responder:

- El modelo mejora en validacion local o solo en leaderboard publico?
- La mejora es estable en varios folds?
- La complejidad adicional justifica el beneficio?
- Puedo reproducir el resultado en una nueva corrida?

Si una mejora no supera estas preguntas, no deberia entrar en tu version final.

### Errores de nivel intermedio que debes evitar

- Ajustar hiperparametros antes de tener un baseline estable.
- Mezclar transformaciones de train y test.
- Hacer demasiados cambios simultaneos.
- Ignorar interpretacion de errores por subgrupo.

### Ejercicio integrador largo

Toma un dataset tabular de Kaggle y completa este ciclo:

1. EDA corto pero accionable.
2. Baseline reproducible.
3. Dos versiones con nuevas features.
4. Una version con tuning moderado.
5. Reporte comparativo final con tabla de metricas.

Objetivo: aprender a pensar como competidor meticuloso, no solo como usuario de librerias.

---

## Navegacion

[← 4. Visualizacion de Datos](/ruta-aprendizaje/4-visualizacion-de-datos) | [6. Fundamentos de Scikit-Learn →](/ruta-aprendizaje/6-fundamentos-de-scikit-learn)
