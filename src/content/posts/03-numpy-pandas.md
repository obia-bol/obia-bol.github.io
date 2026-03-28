---
title: "3. Manejo de Datos con NumPy y Pandas"
description: "Arreglos vectorizados, DataFrames, limpieza, combinacion de datos y extraccion de caracteristicas para problemas reales."
pubDate: "May 03 2026"
badge: "Fase 1"
tags: ["NumPy", "Pandas", "Datos"]
---

## Teoria

El trabajo de IA comienza en los datos. Conceptos clave:

- NumPy arrays: estructura base para computo numerico.
- Broadcasting y operaciones vectorizadas: rendimiento sin loops innecesarios.
- Pandas DataFrames: tablas, indices y columnas.
- Filtrado, agrupacion y agregacion.
- Manejo de faltantes: imputacion, eliminacion y flags.
- Merge y join de datasets.
- Feature extraction basica.
- Introduccion a series temporales.

## Guia practica extensa de NumPy y Pandas

En competencias de IA, la mayor parte del tiempo no se va en "entrenar modelos", sino en preparar datos bien. Esta guia esta enfocada en habilidades de alto impacto: velocidad, claridad y reproducibilidad.

### Parte A: NumPy desde cero hasta uso competitivo

### 1) Creacion de arrays y tipos

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([1.0, 2.0, 3.0])
c = np.zeros((2, 3))
d = np.ones((3, 2))
e = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
f = np.linspace(0, 1, 5)      # 5 puntos entre 0 y 1

print(a.dtype, b.dtype)
print(c.shape, d.shape)
```

Puntos clave:

- `dtype` define precision y memoria.
- `shape` te dice dimensiones.
- Siempre revisa dimensiones antes de operar.

### 2) Casting y control de memoria

```python
x = np.array([1, 2, 3], dtype=np.int64)
print(x.dtype)

x32 = x.astype(np.int32)
print(x32.dtype)

pesos = np.array([0.12, 0.35, 0.88], dtype=np.float64)
pesos32 = pesos.astype(np.float32)
```

En datasets grandes, pasar de `float64` a `float32` puede reducir memoria a la mitad.

### 3) Indexado y slicing

```python
m = np.array([
	[10, 20, 30],
	[40, 50, 60],
	[70, 80, 90]
])

print(m[0, 1])      # 20
print(m[:, 0])      # primera columna
print(m[1:, 1:])    # submatriz
```

Errores comunes:

- Confundir filas con columnas.
- Hacer slicing sin revisar `shape` del resultado.

### 4) Mascaras booleanas

```python
valores = np.array([5, 12, 7, 20, 3, 15])
mask = valores > 10

print(mask)             # [False  True False  True False  True]
print(valores[mask])    # [12 20 15]
```

Esto es fundamental para filtrar rapido sin loops.

### 5) Broadcasting

```python
X = np.array([
	[1.0, 10.0, 100.0],
	[2.0, 20.0, 200.0],
	[3.0, 30.0, 300.0],
])

media = X.mean(axis=0)
X_centrada = X - media

print(media)
print(X_centrada)
```

`X - media` funciona por broadcasting, aplicando el vector a cada fila.

### 6) Operaciones vectorizadas

```python
# Version lenta con loop
nums = np.arange(1, 1_000_001)

out_loop = np.empty_like(nums)
for i, v in enumerate(nums):
	out_loop[i] = v * 2 + 1

# Version vectorizada
out_vec = nums * 2 + 1
```

En IA competitiva, la version vectorizada casi siempre gana en tiempo y claridad.

### 7) Agregaciones por eje

```python
Z = np.array([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9]
])

print(Z.sum())          # suma total
print(Z.sum(axis=0))    # suma por columna
print(Z.sum(axis=1))    # suma por fila
print(Z.mean(axis=0))
```

### 8) Valores faltantes con NumPy

```python
arr = np.array([1.0, np.nan, 3.5, np.nan, 5.0])

print(np.isnan(arr))
print(np.nanmean(arr))

arr_filled = np.where(np.isnan(arr), np.nanmean(arr), arr)
print(arr_filled)
```

### Parte B: Pandas para flujo real de competencia

### 1) Cargar y explorar un dataset

```python
import pandas as pd

df = pd.read_csv("train.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all"))
```

Antes de modelar, responde:

- Cuantas filas y columnas hay?
- Que columnas son numericas/categoricas?
- Cuantos faltantes existen por columna?

### 2) Seleccion de columnas y filas

```python
# Seleccion de columnas
target = df["SalePrice"]
features = df[["LotArea", "OverallQual", "YearBuilt"]]

# loc: por etiqueta
subset_loc = df.loc[df["OverallQual"] >= 7, ["OverallQual", "SalePrice"]]

# iloc: por posicion
subset_iloc = df.iloc[0:5, 0:3]
```

### 3) Filtrado y ordenamiento

```python
filtro = (df["OverallQual"] >= 7) & (df["GrLivArea"] > 1500)
top = df[filtro].sort_values("SalePrice", ascending=False)

print(top[["OverallQual", "GrLivArea", "SalePrice"]].head(10))
```

### 4) Faltantes: detectar e imputar

```python
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(10))

# Imputacion numerica por mediana
for col in ["LotFrontage", "MasVnrArea"]:
	if col in df.columns:
		df[col] = df[col].fillna(df[col].median())

# Imputacion categorica por moda
for col in ["Electrical", "KitchenQual"]:
	if col in df.columns:
		moda = df[col].mode(dropna=True)
		if len(moda) > 0:
			df[col] = df[col].fillna(moda.iloc[0])
```

Practica recomendada: crear una columna bandera para faltantes importantes.

```python
if "LotFrontage" in df.columns:
	df["LotFrontage_was_missing"] = df["LotFrontage"].isna().astype(int)
```

### 5) GroupBy y agregacion

```python
resumen = (
	df.groupby("Neighborhood", dropna=False)
	  .agg(
		  precio_medio=("SalePrice", "mean"),
		  precio_mediana=("SalePrice", "median"),
		  n=("SalePrice", "count")
	  )
	  .sort_values("precio_medio", ascending=False)
)

print(resumen.head(10))
```

### 6) Merge y Join sin perder control

```python
clientes = pd.DataFrame({
	"id": [1, 2, 3],
	"ciudad": ["La Paz", "Cochabamba", "Santa Cruz"]
})

compras = pd.DataFrame({
	"id": [1, 1, 2, 4],
	"monto": [100, 80, 120, 50]
})

left_merge = clientes.merge(compras, on="id", how="left")
inner_merge = clientes.merge(compras, on="id", how="inner")

print(left_merge)
print(inner_merge)
```

Checklist al hacer merge:

- Llaves unicas o duplicadas?
- Tipo de join correcto (`left`, `inner`, `right`, `outer`)?
- Cuantas filas antes y despues?

### 7) Feature engineering con Pandas

```python
df = df.assign(
	HouseAge=lambda x: 2026 - x["YearBuilt"],
	TotalArea=lambda x: x["GrLivArea"] + x.get("TotalBsmtSF", 0),
)

df["is_new_house"] = (df["YearBuilt"] >= 2010).astype(int)
```

Tambien puedes usar transformaciones logaritmicas en variables sesgadas:

```python
if "SalePrice" in df.columns:
	df["SalePrice_log"] = np.log1p(df["SalePrice"])
```

### 8) Series temporales basicas en Pandas

```python
ts = pd.DataFrame({
	"fecha": pd.date_range("2026-01-01", periods=8, freq="D"),
	"ventas": [20, 23, 19, 30, 28, 27, 35, 40]
})

ts["fecha"] = pd.to_datetime(ts["fecha"])
ts = ts.sort_values("fecha")
ts["lag_1"] = ts["ventas"].shift(1)
ts["rolling_3"] = ts["ventas"].rolling(3).mean()

print(ts)
```

### 9) Pipeline tabular simple y reproducible

```python
def preprocess_tabular(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()

	# Limpiar nombres de columnas
	out.columns = [c.strip().replace(" ", "_") for c in out.columns]

	# Quitar duplicados exactos
	out = out.drop_duplicates()

	# Imputar faltantes numericos con mediana
	num_cols = out.select_dtypes(include=["number"]).columns
	for c in num_cols:
		out[c] = out[c].fillna(out[c].median())

	# Imputar faltantes categoricos con "unknown"
	cat_cols = out.select_dtypes(include=["object", "category", "bool"]).columns
	for c in cat_cols:
		out[c] = out[c].fillna("unknown")

	return out
```

Este tipo de funcion te da consistencia entre entrenamiento e inferencia.

## Por que importa en competencias de IA

Un pipeline de datos limpio y rapido puede darte mas ventaja que ajustar hiperparametros durante horas.

## Recursos recomendados

- [**Documentacion oficial de NumPy**](https://numpy.org/doc/stable/): referencia completa de arrays, operaciones y broadcasting
- [**Documentacion oficial de Pandas**](https://pandas.pydata.org/docs/): guia completa de DataFrames, IO y transformaciones
- [**Kaggle Learn — Pandas**](https://www.kaggle.com/learn/pandas): curso interactivo gratuito con ejercicios reales
- [**Kaggle Learn — Data Cleaning**](https://www.kaggle.com/learn/data-cleaning): tecnicas de limpieza aplicadas a datasets reales
- [**Kaggle — Dataset Titanic**](https://www.kaggle.com/competitions/titanic): dataset clasico para practicar EDA y manipulacion de datos

## Ejercicios practicos

- Cargar 2 CSV, unirlos y generar un dataset final.
- Encontrar outliers y valores faltantes.
- Crear 10 features nuevas a partir de columnas existentes.

## Mini-proyectos

- Analisis completo de un dataset real de Kaggle: limpieza, EDA y exportacion.
- Construir una funcion reusable de preprocessing tabular.

### Como resolver los mini-proyectos (resumen practico)

#### Mini-proyecto 1: Analisis completo de un dataset de Kaggle

Objetivo: construir una base de datos limpia y lista para modelar.

Pasos sugeridos:

1. Cargar dataset y revisar estructura (`shape`, `info`, `describe`).
2. Identificar faltantes, duplicados y columnas con alta cardinalidad.
3. Limpiar datos: imputacion, tipos correctos, normalizacion de texto.
4. Hacer EDA corto pero accionable (distribuciones y correlaciones).
5. Crear al menos 5 features nuevas justificadas.
6. Exportar dataset final a `clean_train.csv` y documentar decisiones.

Salida minima esperada:

- Un notebook limpio.
- Un CSV preprocesado.
- Un resumen de decisiones de limpieza y feature engineering.

#### Mini-proyecto 2: Funcion reusable de preprocessing tabular

Objetivo: encapsular limpieza en una sola funcion reutilizable.

Pasos sugeridos:

1. Definir firma: `def preprocess_tabular(df):`.
2. Incluir pasos estandar: columnas, duplicados, faltantes, tipos.
3. Evitar transformar la variable objetivo por accidente.
4. Probar funcion con 2 datasets distintos.
5. Guardar pruebas minimas para validar que no rompe columnas.

Version base recomendada:

```python
def preprocess_tabular(df, target_col=None):
	out = df.copy()

	out.columns = [c.strip().replace(" ", "_") for c in out.columns]
	out = out.drop_duplicates()

	feature_cols = [c for c in out.columns if c != target_col]

	num_cols = out[feature_cols].select_dtypes(include=["number"]).columns
	for c in num_cols:
		out[c] = out[c].fillna(out[c].median())

	cat_cols = out[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns
	for c in cat_cols:
		out[c] = out[c].fillna("unknown")

	return out
```

Consejo de competencia: una buena funcion de preprocessing evita fugas de datos, acelera iteraciones y reduce errores humanos.

## Errores comunes

- Usar loops donde se puede vectorizar.
- Perder el control de indices al hacer merge.
- Tratar faltantes sin justificar la estrategia.

## Seccion avanzada (opcional)

Optimiza memoria con dtypes, categoricas y procesamiento por chunks para datasets grandes.

## Ruta sugerida

1. NumPy basico.
2. Pandas para carga y limpieza.
3. Feature engineering inicial.
4. Pipeline reproducible para modelado.

## Desarrollo extendido para estudio profundo

### Que debes comprender de verdad

No basta con leer definiciones. En este tema debes llegar a tres niveles de dominio:

1. Nivel conceptual: explicar el tema sin mirar apuntes.
2. Nivel tecnico: implementar lo aprendido en codigo o en ejercicios formales.
3. Nivel estrategico: decidir cuando usar esta herramienta en una competencia.

Una forma util de estudiar es la secuencia "leer -> resumir -> implementar -> explicar". Si no puedes explicar una idea con palabras simples, todavia no la dominaste.

### Aplicacion paso a paso en un entorno de competencia

1. Define objetivo y metrica antes de tocar el modelo.
2. Prepara un baseline pequeno y completamente reproducible.
3. Evalua con separacion correcta de datos.
4. Registra decisiones y resultados por experimento.
5. Repite solo cambios pequenos para aislar el impacto.

Este flujo te entrena para competir bajo presion sin perder rigor metodologico.

### Checklist de dominio minimo

- Puedo describir el concepto central del tema con mis palabras.
- Puedo resolver un ejercicio basico sin ayuda externa.
- Puedo identificar al menos dos errores frecuentes del tema.
- Puedo conectar este tema con uno anterior de la ruta.
- Puedo escribir una implementacion limpia y comentada.

### Autoevaluacion sugerida

Responde por escrito:

- Cual es la idea principal del tema y por que importa.
- Que decisiones tomarias al aplicarlo en un problema real.
- Que senales te indican que estas aplicando mal el enfoque.
- Como validarias que tu solucion funciona de verdad.

### Plan de practica de 7 dias

- Dia 1: lectura completa + resumen personal.
- Dia 2: ejercicios basicos.
- Dia 3: ejercicios intermedios.
- Dia 4: mini-proyecto parte 1.
- Dia 5: mini-proyecto parte 2.
- Dia 6: analisis de errores + refactor.
- Dia 7: presentacion breve de resultados.

---

## Navegacion

[← 2. Matematicas para Machine Learning](/ruta-aprendizaje/2-matematicas-para-machine-learning) | [4. Visualizacion de Datos →](/ruta-aprendizaje/4-visualizacion-de-datos)
