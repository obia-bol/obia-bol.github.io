---
title: "2. Matematicas para Machine Learning"
description: "Algebra, calculo basico, algebra lineal y probabilidad como base para entender redes, optimizacion y modelos."
pubDate: "May 02 2026"
badge: "Fase 1"
tags: ["Matematicas", "ML", "Probabilidad"]
---

## Teoria

Para competir en IA no necesitas ser matematico puro, pero si manejar conceptos funcionales:

- Algebra: ecuaciones, funciones, logaritmos y exponenciales.
- Derivadas: intuicion de pendiente y optimizacion.
- Derivadas parciales: funciones con multiples variables.
- Algebra lineal: vectores, matrices, producto punto, multiplicacion matricial.
- Eigenvalores/eigenvectores (intuicion): direcciones importantes en transformaciones.
- Probabilidad: eventos, independencia, Bayes.
- Estadistica: media, varianza, desviacion estandar.

## Guia practica con ejemplos paso a paso

Esta seccion te muestra como usar las matematicas en problemas reales de IA, no solo como teoria abstracta.

### 1) Algebra y funciones

Si tienes una funcion lineal:

<div class="math-block">
	<math display="block">
		<mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>3</mn>
	</math>
</div>

entonces:

- f(0) = 3
- f(2) = 7
- f(-1) = 1

En ML, funciones como esta aparecen en modelos de regresion lineal, donde cambias parametros para ajustar predicciones.

Ejemplo rapido de logaritmos y exponenciales:

<div class="math-block">
	<math display="block">
		<msub><mi>log</mi><mn>10</mn></msub><mo>(</mo><mn>1000</mn><mo>)</mo><mo>=</mo><mn>3</mn>
		<mspace width="1em" />
		<msup><mi>e</mi><mrow><mi>ln</mi><mo>(</mo><mn>5</mn><mo>)</mo></mrow></msup><mo>=</mo><mn>5</mn>
	</math>
</div>

Esto es util cuando aplicas transformaciones logaritmicas para reducir asimetria en datos de precios o ingresos.

### 2) Derivadas y optimizacion

Si:

<div class="math-block">
	<math display="block">
		<mi>g</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup>
	</math>
</div>

su derivada es:

<div class="math-block">
	<math display="block">
		<msup><mi>g</mi><mo>&prime;</mo></msup><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mn>2</mn><mi>x</mi>
	</math>
</div>

Interpretacion:

- En x = 3, la pendiente es 6 (la funcion crece rapido).
- En x = 0, la pendiente es 0 (punto minimo local).

En entrenamiento de modelos, la derivada indica en que direccion mover parametros para bajar el error.

### 3) Derivadas parciales

Si una funcion depende de varias variables, por ejemplo:

<div class="math-block">
	<math display="block">
		<mi>h</mi><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>3</mn><mi>y</mi>
	</math>
</div>

entonces:

<div class="math-block">
	<math display="block">
		<mfrac><mrow><mi>&part;</mi><mi>h</mi></mrow><mrow><mi>&part;</mi><mi>x</mi></mrow></mfrac><mo>=</mo><mn>2</mn><mi>x</mi>
		<mspace width="1em" />
		<mfrac><mrow><mi>&part;</mi><mi>h</mi></mrow><mrow><mi>&part;</mi><mi>y</mi></mrow></mfrac><mo>=</mo><mn>3</mn>
	</math>
</div>

Esto aparece directamente en redes neuronales, donde la perdida depende de muchos pesos.

### 4) Vectores, matrices y producto punto

Dos vectores:

<div class="math-block">
	<math display="block">
		<mi>a</mi><mo>=</mo><mo>[</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>]</mo>
		<mspace width="1em" />
		<mi>b</mi><mo>=</mo><mo>[</mo><mn>4</mn><mo>,</mo><mn>5</mn><mo>,</mo><mn>6</mn><mo>]</mo>
	</math>
</div>

producto punto:

<div class="math-block">
	<math display="block">
		<mi>a</mi><mo>&sdot;</mo><mi>b</mi><mo>=</mo><mn>1</mn><mo>&sdot;</mo><mn>4</mn><mo>+</mo><mn>2</mn><mo>&sdot;</mo><mn>5</mn><mo>+</mo><mn>3</mn><mo>&sdot;</mo><mn>6</mn><mo>=</mo><mn>32</mn>
	</math>
</div>

El producto punto mide "alineacion" entre vectores y se usa en similitud, embeddings y capas lineales.

Multiplicacion de matrices (ejemplo corto):

<div class="math-block">
	<math display="block">
		<mi>A</mi><mo>=</mo>
		<mfenced open="[" close="]">
			<mtable>
				<mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr>
				<mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr>
			</mtable>
		</mfenced>
		<mspace width="1em" />
		<mi>B</mi><mo>=</mo>
		<mfenced open="[" close="]">
			<mtable>
				<mtr><mtd><mn>2</mn></mtd><mtd><mn>0</mn></mtd></mtr>
				<mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr>
			</mtable>
		</mfenced>
		<mspace width="1em" />
		<mi>AB</mi><mo>=</mo>
		<mfenced open="[" close="]">
			<mtable>
				<mtr><mtd><mn>4</mn></mtd><mtd><mn>4</mn></mtd></mtr>
				<mtr><mtd><mn>10</mn></mtd><mtd><mn>8</mn></mtd></mtr>
			</mtable>
		</mfenced>
	</math>
</div>

### 5) Eigenvalores (intuicion)

Cuando aplicas una transformacion lineal, hay direcciones que solo cambian de escala. Esas direcciones son eigenvectores, y el factor de escala es el eigenvalor.

En PCA, estas ideas te ayudan a encontrar las direcciones con mayor varianza para reducir dimensionalidad sin perder demasiada informacion.

### 6) Probabilidad y Teorema de Bayes

Bayes:

<div class="math-block">
	<math display="block">
		<mi>P</mi><mo>(</mo><mi>A</mi><mo>|</mo><mi>B</mi><mo>)</mo><mo>=</mo>
		<mfrac>
			<mrow><mi>P</mi><mo>(</mo><mi>B</mi><mo>|</mo><mi>A</mi><mo>)</mo><mi>P</mi><mo>(</mo><mi>A</mi><mo>)</mo></mrow>
			<mrow><mi>P</mi><mo>(</mo><mi>B</mi><mo>)</mo></mrow>
		</mfrac>
	</math>
</div>

Ejemplo medico sencillo:

- P(Enfermo) = 0.01
- P(Test+|Enfermo) = 0.95
- P(Test+|Sano) = 0.05

Primero:

<div class="math-block">
	<math display="block">
		<mi>P</mi><mo>(</mo><mi>Test</mi><mo>+</mo><mo>)</mo><mo>=</mo><mn>0.95</mn><mo>&sdot;</mo><mn>0.01</mn><mo>+</mo><mn>0.05</mn><mo>&sdot;</mo><mn>0.99</mn><mo>=</mo><mn>0.059</mn>
	</math>
</div>

Luego:

<div class="math-block">
	<math display="block">
		<mi>P</mi><mo>(</mo><mi>Enfermo</mi><mo>|</mo><mi>Test</mi><mo>+</mo><mo>)</mo><mo>=</mo>
		<mfrac><mrow><mn>0.95</mn><mo>&sdot;</mo><mn>0.01</mn></mrow><mrow><mn>0.059</mn></mrow></mfrac>
		<mo>&approx;</mo><mn>0.161</mn>
	</math>
</div>

Aunque el test sea bueno, la probabilidad posterior puede no ser tan alta si la prevalencia es baja.

### 7) Estadistica descriptiva minima

Para datos [2,4,4,4,5,5,7,9]:

- Media: 5
- Varianza (poblacional): 4
- Desviacion estandar: 2

Estas medidas son basicas para EDA, estandarizacion y deteccion de outliers.

## Codigo Python para reforzar matematicas

```python
import numpy as np

# Producto punto
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Producto punto:", np.dot(a, b))  # 32

# Multiplicacion de matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
print("A @ B:\n", A @ B)

# Estadistica basica
datos = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print("Media:", np.mean(datos))
print("Varianza:", np.var(datos))
print("Desv. estandar:", np.std(datos))
```

Si puedes conectar estos calculos con intuicion, estas construyendo una base solida para modelos mas avanzados.

## Por que importa en competencias de IA

Ayuda a entender funciones de perdida, gradiente, PCA, regularizacion y comportamiento de modelos.

## Recursos recomendados

- [**3Blue1Brown — Algebra Lineal**](https://www.3blue1brown.com/topics/linear-algebra): la mejor visualizacion geometrica de vectores, matrices y transformaciones
- [**3Blue1Brown — Redes Neuronales**](https://www.3blue1brown.com/topics/neural-networks): intuicion sobre gradientes y backpropagation con animaciones
- [**Khan Academy — Calculo**](https://www.khanacademy.org/math/calculus-1): derivadas e integrales desde cero, completamente gratuito
- [**Khan Academy — Estadistica y Probabilidad**](https://www.khanacademy.org/math/statistics-probability): distribuciones, inferencia y probabilidad condicional
- [**Mathematics for Machine Learning (Coursera)**](https://www.coursera.org/specializations/mathematics-machine-learning): especialidad completa que conecta matematicas con ML

## Ejercicios practicos

- Calcular a mano media, varianza y desviacion de un conjunto pequeno.
- Resolver ejemplos de Bayes para clasificacion binaria.
- Programar multiplicacion de matrices y producto punto en Python.

## Mini-proyectos

- Notebook explicando descenso de gradiente en 1D y 2D con graficas.
- Implementacion simple de PCA desde cero para datos pequenos.

### Como resolver los mini-proyectos (resumen practico)

#### Mini-proyecto 1: Descenso de gradiente en 1D y 2D

Objetivo: entender como los parametros se actualizan para minimizar una funcion de costo.

Pasos sugeridos:

1. Define una funcion simple, por ejemplo f(x) = x^2 (1D).
2. Deriva la funcion: f'(x) = 2x.
3. Elige un punto inicial (por ejemplo x = 8) y learning rate (por ejemplo 0.1).
4. Actualiza: x_nuevo = x_actual - eta \* f'(x).
5. Repite por iteraciones y grafica como baja el valor de f(x).
6. Para 2D, usa una funcion tipo f(x, y) = x^2 + y^2 y aplica gradiente por componente.

Resultados esperados:

- Ver convergencia al minimo.
- Entender efecto de learning rate pequeno vs grande.

#### Mini-proyecto 2: PCA desde cero (version educativa)

Objetivo: reducir dimensionalidad preservando informacion relevante.

Pasos sugeridos:

1. Estandariza los datos (restar media y dividir por desviacion).
2. Calcula la matriz de covarianza.
3. Obtiene eigenvalores y eigenvectores.
4. Ordena componentes por eigenvalor descendente.
5. Proyecta datos en los primeros componentes.
6. Grafica datos originales vs proyectados.

Resultados esperados:

- Comprender por que PCA conserva direcciones de mayor varianza.
- Poder justificar cuantos componentes conservar.

Consejo de competencia: en datasets tabulares con muchas variables correlacionadas, PCA puede ayudar a simplificar modelos y acelerar entrenamiento.

## Preguntas de auto-chequeo rapido

- Puedo explicar por que el gradiente apunta hacia el aumento mas rapido y por que usamos su negativo?
- Puedo calcular un producto punto sin calculadora en un caso pequeno?
- Puedo interpretar la salida de media/varianza en contexto de datos reales?
- Puedo describir PCA sin decir solo "reduce dimensiones"?

## Errores comunes

- Memorizar formulas sin intuicion.
- Ignorar escalado de variables.
- No conectar teoria con codigo y experimentos.

## Seccion avanzada (opcional)

Revisa descomposicion SVD y relacion con reduccion de dimensionalidad.

## Ruta sugerida

1. Estadistica descriptiva.
2. Probabilidad y Bayes.
3. Algebra lineal aplicada.
4. Derivadas y optimizacion.
5. Aplicacion en notebooks de ML.

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

[← 1. Fundamentos de Programacion en Python](/ruta-aprendizaje/1-fundamentos-de-programacion-en-python) | [3. Manejo de Datos con NumPy y Pandas →](/ruta-aprendizaje/3-manejo-de-datos-con-numpy-y-pandas)
