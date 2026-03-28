---
title: "1. Fundamentos de Programacion en Python"
description: "Variables, control de flujo, funciones, estructuras de datos, archivos, OOP basica y manejo de entornos con pip."
pubDate: "May 01 2026"
badge: "Fase 1"
tags: ["Python", "Fundamentos", "Practica"]
---

## Teoria

Python es el lenguaje base en la mayoria de rutas competitivas de IA. Debes dominar:

- Variables y tipos: int, float, str, bool.
- Condicionales y bucles: if/elif/else, for, while.
- Funciones: parametros, retorno, scope.
- Estructuras: listas, diccionarios, sets, tuplas.
- Archivos: lectura/escritura de txt y csv.
- Clases y OOP basica: atributos, metodos, encapsulacion simple.
- Entornos virtuales: venv para aislar dependencias.
- Pip y paquetes: instalacion, versionado y buenas practicas.

## Guia practica con codigo

Esta seccion muestra ejemplos minimos y claros de lo esencial que debes saber para comenzar.

### 1) Variables y tipos basicos

```python
# Declaracion de variables
edad = 16              # int
altura = 1.72          # float
nombre = "Ana"         # str
es_estudiante = True   # bool

print(type(edad))          # <class 'int'>
print(type(altura))        # <class 'float'>
print(type(nombre))        # <class 'str'>
print(type(es_estudiante)) # <class 'bool'>
```

### 2) Conversion de tipos (casting)

```python
numero_texto = "42"
numero_int = int(numero_texto)        # str -> int
numero_float = float(numero_texto)    # str -> float

precio = 19.99
precio_entero = int(precio)           # float -> int (trunca decimales)

valor = 7
valor_str = str(valor)                # int -> str

print(numero_int, numero_float, precio_entero, valor_str)
```

### 3) Condicionales con if / elif / else

```python
nota = 78

if nota >= 90:
	print("Excelente")
elif nota >= 70:
	print("Aprobado")
else:
	print("Debes reforzar")
```

### 4) Como declarar funciones y usar return

```python
def calcular_promedio(a, b, c):
	promedio = (a + b + c) / 3
	return promedio

resultado = calcular_promedio(80, 75, 90)
print("Promedio:", resultado)
```

Notas importantes:

- `def` declara una funcion.
- `return` devuelve un valor para usarlo fuera de la funcion.
- Una funcion sin `return` devuelve `None`.

### 5) Entrada por teclado con input()

```python
nombre = input("Ingresa tu nombre: ")
edad = int(input("Ingresa tu edad: "))

print(f"Hola {nombre}, el proximo anio tendras {edad + 1}.")
```

Recuerda: `input()` siempre devuelve texto, por eso se usa `int()` o `float()` cuando corresponde.

### 6) Leer desde un archivo .txt

Supongamos un archivo llamado `datos.txt` con este contenido:

```text
15
20
35
```

Codigo para leerlo y sumar valores:

```python
suma = 0

with open("datos.txt", "r", encoding="utf-8") as archivo:
	for linea in archivo:
		numero = int(linea.strip())
		suma += numero

print("Suma total:", suma)
```

### 7) Ejemplo simple integrador

Este ejemplo combina tipos, input, if, funciones y lectura de archivo.

```python
def clasificar_puntaje(puntaje):
	if puntaje >= 85:
		return "Nivel alto"
	elif puntaje >= 60:
		return "Nivel medio"
	return "Nivel inicial"

nombre = input("Nombre del estudiante: ")
puntaje = int(input("Puntaje del examen: "))

nivel = clasificar_puntaje(puntaje)
print(f"{nombre} -> {nivel}")

with open("reporte.txt", "w", encoding="utf-8") as salida:
	salida.write(f"Estudiante: {nombre}\n")
	salida.write(f"Puntaje: {puntaje}\n")
	salida.write(f"Clasificacion: {nivel}\n")
```

Con este programa ya practicas operaciones clave para competencias y tareas de preprocesamiento de datos.

## Por que importa en competencias de IA

La velocidad para prototipar soluciones en Python impacta directamente en tu desempeno bajo tiempo limitado.

## Recursos recomendados

- [**Documentacion oficial de Python**](https://docs.python.org/3/tutorial/): tutorial completo del lenguaje, disponible en espanol
- [**Real Python**](https://realpython.com/): articulos y tutoriales sobre fundamentos, OOP y buenas practicas
- [**Kaggle Learn — Python**](https://www.kaggle.com/learn/python): curso interactivo gratuito con notebooks ejecutables
- [**Kaggle Learn — Intro to Machine Learning**](https://www.kaggle.com/learn/intro-to-machine-learning): siguiente paso natural despues de los fundamentos

## Ejercicios practicos

- Resolver 30 ejercicios de nivel basico/intermedio.
- Parsear un CSV y calcular estadisticos simples por columna.
- Implementar funciones reutilizables para limpieza de datos.

## Mini-proyectos

- Construir un analizador de dataset CSV desde consola.
- Crear una clase `ExperimentLogger` que guarde resultados en JSON.

### Como resolver los mini-proyectos (resumen practico)

#### Mini-proyecto 1: Analizador de CSV desde consola

Objetivo: leer un CSV, mostrar informacion general y calcular estadisticos basicos.

Pasos sugeridos:

1. Recibir el nombre del archivo con `input()`.
2. Abrir el archivo con `open(...)` o usar `csv.DictReader`.
3. Contar filas y columnas.
4. Para columnas numericas, calcular minimo, maximo y promedio.
5. Mostrar resultados en consola de forma ordenada.

Estructura minima:

```python
import csv

archivo = input("Nombre del CSV: ")

with open(archivo, "r", encoding="utf-8") as f:
	reader = csv.DictReader(f)
	filas = list(reader)

print("Filas:", len(filas))
print("Columnas:", reader.fieldnames)
```

#### Mini-proyecto 2: Clase ExperimentLogger en JSON

Objetivo: guardar resultados de experimentos para comparar modelos.

Pasos sugeridos:

1. Crear clase `ExperimentLogger`.
2. Guardar cada experimento como diccionario: nombre, metrica, fecha, notas.
3. Acumular en lista y exportar a JSON.
4. Cargar JSON para revisar historial.

Estructura minima:

```python
import json
from datetime import datetime

class ExperimentLogger:
	def __init__(self):
		self.logs = []

	def add(self, nombre, metrica, notas=""):
		self.logs.append({
			"nombre": nombre,
			"metrica": metrica,
			"notas": notas,
			"fecha": datetime.now().isoformat()
		})

	def save(self, ruta="experimentos.json"):
		with open(ruta, "w", encoding="utf-8") as f:
			json.dump(self.logs, f, indent=2, ensure_ascii=False)
```

Esto te prepara para competencias donde necesitas comparar versiones de forma disciplinada.

## Errores comunes

- Mezclar logica y entrada/salida en la misma funcion.
- No usar entornos virtuales y romper dependencias.
- Escribir codigo sin modularidad.

## Seccion avanzada (opcional)

Aprende comprensiones, decoradores basicos y tipado con `typing` para mejorar legibilidad y mantenibilidad.

## Ruta sugerida

1. Sintaxis base y estructuras.
2. Funciones y organizacion modular.
3. Archivos y parsing de datos.
4. OOP minima para proyectos.
5. Entorno virtual y gestion de paquetes.

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

[← 0. Introduccion a las Olimpiadas de IA](/ruta-aprendizaje/0-introduccion-a-las-olimpiadas-de-ia) | [2. Matematicas para Machine Learning →](/ruta-aprendizaje/2-matematicas-para-machine-learning)
