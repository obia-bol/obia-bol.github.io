---
title: "6. Fundamentos de Scikit-Learn"
description: "Workflow de ML con pipelines, fit/predict/transform, validacion cruzada y tuning de hiperparametros con GridSearchCV."
pubDate: "May 06 2026"
badge: "Fase 2"
tags: ["Scikit-learn", "Pipelines", "Cross-validation"]
---

## Teoria

Scikit-learn es una libreria estandar para ML clasico, muy usada en IOAI.

Conceptos obligatorios:

- Flujo de trabajo ML de punta a punta.
- `fit`, `predict`, `transform`.
- Pipelines para evitar leakage.
- Cross-validation.
- Hyperparameter tuning con `GridSearchCV`.
- Seleccion de metricas segun objetivo.

## Guia practica extensa de Scikit-Learn

Esta guia te muestra como usar scikit-learn en flujo real de competencia: rapido, ordenado y reproducible.

### 1) Patrón base: fit, predict, transform

En scikit-learn casi todo sigue este patron:

- `fit`: aprende parametros desde datos.
- `transform`: transforma datos (escalado, encoding, etc.).
- `predict`: produce predicciones.

Ejemplo corto:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform en train
X_test_scaled = scaler.transform(X_test)        # solo transform en test

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)
pred = clf.predict(X_test_scaled)
```

### 2) Pipeline: la forma correcta de trabajar

Pipeline evita leakage y hace tu código más limpio.

```python
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

pipe = Pipeline([
	("scaler", StandardScaler()),
	("model", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print("F1:", f1_score(y_test, pred))
```

### 3) Datos mixtos: ColumnTransformer

En tabulares reales tienes columnas numericas y categoricas. Usa `ColumnTransformer`.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
	"edad": [18, 25, None, 41, 33],
	"ingreso": [1000, 2000, 1500, None, 3000],
	"ciudad": ["A", "B", "A", "C", None],
	"target": [0, 1, 0, 1, 1],
})

X = df.drop(columns=["target"])
y = df["target"]

num_cols = ["edad", "ingreso"]
cat_cols = ["ciudad"]

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

full_model = Pipeline([
	("prep", preprocess),
	("clf", LogisticRegression(max_iter=1000))
])

full_model.fit(X, y)
```

### 4) Cross-validation: comparar modelos con justicia

No confies en una sola particion train/test.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

print("Scores por fold:", scores)
print("F1 medio:", scores.mean())
```

### 5) KFold vs StratifiedKFold

- `KFold`: divide en partes iguales sin cuidar balance de clases.
- `StratifiedKFold`: mantiene proporciones de clases en cada fold.

Si tu problema es de clasificacion con clases desbalanceadas, usa `StratifiedKFold` casi siempre.

### 6) GridSearchCV para tuning ordenado

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
	"model__C": [0.01, 0.1, 1, 10],
	"model__penalty": ["l2"],
}

grid = GridSearchCV(
	estimator=pipe,
	param_grid=param_grid,
	scoring="f1",
	cv=cv,
	n_jobs=-1,
	verbose=0
)

grid.fit(X_train, y_train)

print("Mejores parametros:", grid.best_params_)
print("Mejor score CV:", grid.best_score_)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)
print("F1 test:", f1_score(y_test, pred))
```

### 7) Leer resultados de tuning

No mires solo `best_score_`. Revisa estabilidad:

```python
results = pd.DataFrame(grid.cv_results_)
cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
print(results[cols].sort_values("rank_test_score").head(10))
```

Si la desviacion entre folds es alta, tu solucion puede ser inestable.

### 8) Metricas segun objetivo

Clasificacion:

- Accuracy cuando clases estan balanceadas.
- F1 cuando hay desbalance moderado.
- ROC-AUC para evaluar ranking de probabilidades.

Regresion:

- MAE: error medio absoluto.
- RMSE: penaliza mas los errores grandes.
- R2: proporcion de varianza explicada.

### 9) Comparar modelos rapidamente

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

models = {
	"logreg": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
	"rf": Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=42))]),
	"svm": Pipeline([("scaler", StandardScaler()), ("model", SVC())])
}

for name, m in models.items():
	score = cross_val_score(m, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
	print(f"{name}: {score:.4f}")
```

### 10) Baseline de competencia en 30-60 minutos

Checklist minimo:

1. Definir metrica correcta.
2. Split reproducible.
3. Pipeline con preprocessing dentro.
4. CV para comparar.
5. Baseline documentado.

Con esto ya puedes iterar con criterio tecnico.

## Por que importa en competencias de IA

Permite iterar rapido, comparar modelos de forma justa y construir soluciones robustas bajo tiempo limitado.

## Recursos recomendados

- [**Documentacion oficial de scikit-learn**](https://scikit-learn.org/stable/): referencia completa con ejemplos para cada modelo y utilidad
- [**User Guide — Model Selection**](https://scikit-learn.org/stable/model_selection.html): validacion cruzada, metricas y busqueda de hiperparametros
- [**User Guide — Pipelines**](https://scikit-learn.org/stable/modules/compose.html): construccion de pipelines reproducibles y sin leakage
- [**Kaggle Learn — Intro to Machine Learning**](https://www.kaggle.com/learn/intro-to-machine-learning): practica con sklearn desde cero
- [**Kaggle Learn — Intermediate Machine Learning**](https://www.kaggle.com/learn/intermediate-machine-learning): pipelines y validacion avanzada

## Ejercicios practicos

- Crear pipeline con escalado + modelo.
- Comparar KFold vs StratifiedKFold.
- Ejecutar GridSearchCV y analizar resultados.

## Mini-proyectos

- Entrenar tus primeros modelos ML en un dataset publico y publicar una bitacora de experimentos.

### Como resolver el mini-proyecto (resumen practico)

Objetivo: entrenar 3 modelos comparables y elegir uno con evidencia.

Pasos sugeridos:

1. Elegir dataset tabular (Titanic, Adult Income o House Prices).
2. Definir metrica principal.
3. Crear pipeline reproducible con preprocessing.
4. Entrenar al menos 3 modelos (ejemplo: Logistic Regression, Random Forest, XGBoost/Gradient Boosting).
5. Evaluar con validacion cruzada consistente.
6. Guardar resultados en una tabla de experimentos.
7. Elegir modelo final y justificar por rendimiento + estabilidad.

Formato sugerido de bitacora:

```text
exp_01 | modelo=logreg | features=v1 | cv_f1=0.742 | notas=baseline
exp_02 | modelo=rf     | features=v1 | cv_f1=0.768 | notas=mejora en recall
exp_03 | modelo=rf     | features=v2 | cv_f1=0.781 | notas=feature engineering
```

Resultado esperado:

- Notebook o script ejecutable.
- Tabla de experimentos clara.
- Decisiones tecnicas justificadas.

## Errores comunes

- Hacer preprocessing fuera del pipeline.
- Elegir metricas que no reflejan el objetivo de negocio/competencia.
- Usar demasiada busqueda de hiperparametros sin una buena validacion.

## Seccion avanzada (opcional)

Explora `RandomizedSearchCV`, `ColumnTransformer` y ensambles basicos.

## Ruta sugerida

1. Pipeline minimo funcional.
2. Validacion cruzada confiable.
3. Tuning progresivo y analisis de errores.

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

[← 5. Introduccion a Machine Learning](/ruta-aprendizaje/5-introduccion-a-machine-learning) | [7. Modelos de Regresion →](/ruta-aprendizaje/7-modelos-de-regresion)
