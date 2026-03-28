---
title: "17. Etica y IA Responsable"
description: "Sesgo, fairness, privacidad, alucinaciones, riesgos de mal uso y despliegue responsable de sistemas de IA."
pubDate: "May 17 2026"
badge: "Fase 5"
tags: ["Etica", "Fairness", "Privacidad", "Responsible AI"]
---

## Por que la etica no es un anexo

Cuando un modelo de credito rechaza a personas de un grupo demografico el doble de veces que a otro, eso no es un fallo tecnico menor: es un dano real sobre personas reales. Cuando un sistema de reconocimiento facial falla 10 veces mas en mujeres con piel oscura que en hombres con piel clara (Buolamwini & Gebru, 2018), el problema no es el accuracy promedio — es que el promedio esconde inequidad.

La etica en IA no es una materia separada de la tecnica. Es la pregunta que debes hacerte antes de definir la metrica: **para quien funciona bien este modelo, y para quien falla?**

---

## 1. Sesgo en datos y modelos

El sesgo (bias) en ML tiene dos origenes principales:

**Sesgo en los datos:**

- _Sesgo de seleccion_: el dataset no representa a la poblacion objetivo. Ej: entrenas deteccion de enfermedades con datos de un hospital privado y despliegas en hospitales publicos.
- _Sesgo de etiquetado_: los anotadores humanos introducen sus propios prejuicios. Ej: anotadores que etiquetan resumenes de hojas de vida como "candidato fuerte" inconsistentemente segun el nombre.
- _Sesgo historico_: el dataset refleja inequidades del pasado. Ej: datos de contratacion donde las mujeres estaban subrepresentadas en puestos tecnicos — el modelo aprende a replicar esa inequidad.

**Sesgo amplificado por el modelo:**

- Los modelos pueden amplificar patrones estadisticos minoritarios. Un sesgo leve en los datos puede volverse un sesgo fuerte en las predicciones.
- La funcion de perdida estandar optimiza el accuracy promedio, que puede mejorar ignorando subgrupos minoritarios.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

def auditoria_sesgo(y_true, y_pred, y_prob, grupo, nombre_grupo="grupo"):
    """
    Audita las predicciones de un modelo por subgrupos.

    Calcula metricas separadas por cada valor del atributo 'grupo'
    y compara contra el rendimiento global.

    Params:
        y_true:      etiquetas reales
        y_pred:      predicciones del modelo (0/1)
        y_prob:      probabilidades (para ROC-AUC)
        grupo:       array con el atributo sensible (ej: genero, edad_grupo)
        nombre_grupo: nombre del atributo para imprimir
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "grupo":  grupo,
    })

    print(f"\n{'='*55}")
    print(f"Auditoria de sesgo por: {nombre_grupo}")
    print(f"{'='*55}")
    print(f"\nGlobal:")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  Tasa positiva: {y_pred.mean():.4f}")

    resultados = {}
    for g in sorted(df["grupo"].unique()):
        mask = df["grupo"] == g
        sub  = df[mask]
        acc  = accuracy_score(sub["y_true"], sub["y_pred"])
        auc  = roc_auc_score(sub["y_true"], sub["y_prob"]) if sub["y_true"].nunique() > 1 else float("nan")
        fpr  = ((sub["y_pred"] == 1) & (sub["y_true"] == 0)).sum() / (sub["y_true"] == 0).sum()
        fnr  = ((sub["y_pred"] == 0) & (sub["y_true"] == 1)).sum() / (sub["y_true"] == 1).sum()
        tasa_pos = sub["y_pred"].mean()
        resultados[g] = {"accuracy": acc, "roc_auc": auc, "FPR": fpr, "FNR": fnr, "tasa_positiva": tasa_pos, "n": len(sub)}
        print(f"\n  Grupo '{g}' (n={len(sub)}):")
        print(f"    Accuracy:       {acc:.4f}")
        print(f"    ROC-AUC:        {auc:.4f}")
        print(f"    FPR:            {fpr:.4f}  (False Positive Rate)")
        print(f"    FNR:            {fnr:.4f}  (False Negative Rate)")
        print(f"    Tasa positiva:  {tasa_pos:.4f}")

    # Disparidad maxima entre grupos
    grupos = list(resultados.keys())
    if len(grupos) >= 2:
        disp_acc = max(resultados[g]["accuracy"] for g in grupos) - min(resultados[g]["accuracy"] for g in grupos)
        disp_fpr = max(resultados[g]["FPR"] for g in grupos) - min(resultados[g]["FPR"] for g in grupos)
        disp_fnr = max(resultados[g]["FNR"] for g in grupos) - min(resultados[g]["FNR"] for g in grupos)
        print(f"\n  Disparidad maxima entre grupos:")
        print(f"    Accuracy: {disp_acc:.4f}", "⚠" if disp_acc > 0.05 else "ok")
        print(f"    FPR:      {disp_fpr:.4f}", "⚠" if disp_fpr > 0.05 else "ok")
        print(f"    FNR:      {disp_fnr:.4f}", "⚠" if disp_fnr > 0.05 else "ok")

    return pd.DataFrame(resultados).T


# Ejemplo simulado
np.random.seed(42)
n = 1000
grupo = np.where(np.random.rand(n) < 0.4, "A", "B")
y_true = np.random.binomial(1, np.where(grupo == "A", 0.4, 0.5), n)
# Modelo con sesgo: peor en grupo A
y_prob = np.where(grupo == "A",
                  np.clip(y_true * 0.6 + np.random.normal(0, 0.2, n), 0.01, 0.99),
                  np.clip(y_true * 0.8 + np.random.normal(0, 0.15, n), 0.01, 0.99))
y_pred = (y_prob > 0.5).astype(int)

resultados_df = auditoria_sesgo(y_true, y_pred, y_prob, grupo, "genero_simulado")
```

---

## 2. Definiciones de Fairness (y sus tensiones)

No existe una definicion unica de "justo". Las mas usadas son matematicamente incompatibles entre si — no se pueden cumplir todas al mismo tiempo en la mayoria de los casos reales (teorema de imposibilidad de Chouldechova, 2017).

| Definicion                                   | Formula                                    | Cuando usarla                                                                     |
| -------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------- |
| **Demographic Parity** (Paridad demografica) | P(ŷ=1\|A=0) = P(ŷ=1\|A=1)                  | Cuando la tasa de positivos debe ser igual entre grupos (ej: publicidad)          |
| **Equal Opportunity**                        | TPR_A = TPR_B                              | Cuando los falsos negativos son el dano principal (ej: diagnostico de enfermedad) |
| **Equalized Odds**                           | TPR_A = TPR_B y FPR_A = FPR_B              | Mas estricto: igual TPR y FPR entre grupos                                        |
| **Predictive Parity**                        | P(y=1\|ŷ=1,A=0) = P(y=1\|ŷ=1,A=1)          | Precision igual entre grupos (ej: scoring de riesgo)                              |
| **Individual Fairness**                      | sim(x_i, x_j) alta → pred(x_i) ≈ pred(x_j) | Individuos similares deben recibir predicciones similares                         |

La **tension fundamental**: si la tasa base de eventos positivos difiere entre grupos (P(y=1|A) ≠ P(y=1|B)), entonces Demographic Parity y Predictive Parity no pueden cumplirse simultaneamente con alta accuracy.

```python
from sklearn.metrics import confusion_matrix

def metricas_fairness(y_true, y_pred, grupos):
    """
    Calcula metricas de fairness estandar para clasificacion binaria.

    Retorna un DataFrame con TPR, FPR, PPV (precision) por grupo
    y los ratios entre grupos (para detectar disparidad).
    """
    resultados = {}
    for g in sorted(set(grupos)):
        mask = grupos == g
        yt, yp = y_true[mask], y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")  # Fall-out
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")  # Precision
        tasa_pos = yp.mean()
        resultados[g] = {"TPR": tpr, "FPR": fpr, "PPV": ppv, "tasa_positiva": tasa_pos, "n": mask.sum()}

    df = pd.DataFrame(resultados).T
    grupos_unicos = list(df.index)
    if len(grupos_unicos) == 2:
        g0, g1 = grupos_unicos
        print("\nRatios de disparidad (valor cercano a 1.0 = equitativo):")
        for metrica in ["TPR", "FPR", "PPV", "tasa_positiva"]:
            r = df.loc[g0, metrica] / df.loc[g1, metrica]
            flag = "⚠  disparidad" if r < 0.8 or r > 1.25 else "ok"
            print(f"  {metrica}: {df.loc[g0, metrica]:.3f} / {df.loc[g1, metrica]:.3f} = {r:.3f}  {flag}")
    return df


# Con IBM AIF360 (instalacion: pip install aif360)
# from aif360.datasets import BinaryLabelDataset
# from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from aif360.algorithms.preprocessing import Reweighing
#
# dataset = BinaryLabelDataset(df=df, label_names=["target"],
#                              protected_attribute_names=["genero"])
# metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{"genero": 1}],
#                                   unprivileged_groups=[{"genero": 0}])
# print("Disparate Impact:", metric.disparate_impact())
# print("Statistical Parity Difference:", metric.statistical_parity_difference())
```

---

## 3. Tecnicas de mitigacion de sesgo

La mitigacion puede aplicarse en tres momentos del pipeline:

### Pre-procesamiento (sobre los datos)

```python
import numpy as np
from sklearn.utils import compute_sample_weight

# ── Reweighting: dar mas peso a los grupos subrepresentados ──────────────────
def calcular_pesos_fairness(y_true, grupos, metodo="reweighting"):
    """
    Calcula sample weights para reducir disparidad demografica.

    metodo="reweighting": ajusta pesos para que cada (grupo, clase) tenga
                          la misma influencia en el entrenamiento.
    """
    df = pd.DataFrame({"y": y_true, "g": grupos})
    n_total = len(df)
    pesos = np.ones(n_total)

    if metodo == "reweighting":
        for (g, y), grupo_df in df.groupby(["g", "y"]):
            n_grupo = (df["g"] == g).sum()
            n_clase = (df["y"] == y).sum()
            n_grupo_clase = len(grupo_df)
            # Peso = P(grupo) * P(clase) / P(grupo, clase)
            peso = (n_grupo / n_total) * (n_clase / n_total) / (n_grupo_clase / n_total)
            pesos[grupo_df.index] = peso

    return pesos


# ── Oversampling de grupos subrepresentados ───────────────────────────────────
def oversample_grupo(X, y, grupos, grupo_minoritario):
    """Duplica muestras del grupo minoritario hasta equilibrar."""
    mask_min = grupos == grupo_minoritario
    mask_max = ~mask_min
    n_max = mask_max.sum()
    n_min = mask_min.sum()
    if n_min >= n_max:
        return X, y, grupos

    factor = n_max // n_min
    idx_min = np.where(mask_min)[0]
    idx_extra = np.tile(idx_min, factor - 1)
    idx_all = np.concatenate([np.arange(len(y)), idx_extra])
    np.random.shuffle(idx_all)

    return X[idx_all], y[idx_all], grupos[idx_all]
```

### In-procesamiento (durante el entrenamiento)

```python
# ── Regularizacion con penalizacion por disparidad ───────────────────────────
# La idea: agregar un termino a la funcion de perdida que penaliza la diferencia
# de tasas de positivos entre grupos.
#
# Loss_total = Loss_clasificacion + lambda * |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
#
# Implementacion personalizada con PyTorch:

import torch
import torch.nn as nn

def perdida_con_fairness(logits, y_true, grupos, lambda_fair=0.5):
    """
    Funcion de perdida con regularizacion de fairness (demographic parity).

    logits: [batch, 1] — salida del modelo antes del sigmoid
    y_true: [batch]    — etiquetas
    grupos: [batch]    — atributo protegido (0 o 1)
    """
    # Perdida de clasificacion estandar
    loss_cls = nn.BCEWithLogitsLoss()(logits.squeeze(), y_true.float())

    # Penalizacion de fairness: diferencia en tasa de positivos entre grupos
    probs = torch.sigmoid(logits.squeeze())
    tasa_0 = probs[grupos == 0].mean()
    tasa_1 = probs[grupos == 1].mean()
    loss_fair = torch.abs(tasa_0 - tasa_1)

    return loss_cls + lambda_fair * loss_fair
```

### Post-procesamiento (sobre las predicciones)

```python
# ── Ajuste de umbral por grupo ────────────────────────────────────────────────
# Si TPR difiere mucho entre grupos, ajustar el umbral de decision
# de forma independiente para cada grupo puede restaurar Equal Opportunity.

def calibrar_umbrales_fairness(y_true, y_prob, grupos, objetivo="equal_opportunity"):
    """
    Encuentra umbrales de clasificacion por grupo para cumplir una metrica de fairness.

    objetivo:
      "equal_opportunity"   -> igualar TPR entre grupos
      "demographic_parity"  -> igualar tasa de positivos entre grupos
    """
    from sklearn.metrics import roc_curve

    umbrales = {}
    metricas_objetivo = {}

    for g in sorted(set(grupos)):
        mask = grupos == g
        yt, yp = y_true[mask], y_prob[mask]
        fpr, tpr, thresholds = roc_curve(yt, yp)

        if objetivo == "equal_opportunity":
            # Guardar TPR en funcion del umbral
            metricas_objetivo[g] = list(zip(tpr, thresholds))
        else:  # demographic_parity
            # Guardar tasa de positivos en funcion del umbral
            tasas = [(yp >= t).mean() for t in thresholds]
            metricas_objetivo[g] = list(zip(tasas, thresholds))

    # Objetivo: todos los grupos tengan la misma metrica que el grupo base
    grupo_base = sorted(set(grupos))[0]
    # Usar el umbral 0.5 como referencia para el grupo base
    mask_base = grupos == grupo_base
    if objetivo == "equal_opportunity":
        preds_base = (y_prob[mask_base] >= 0.5).astype(int)
        ref = ((preds_base == 1) & (y_true[mask_base] == 1)).sum() / (y_true[mask_base] == 1).sum()
    else:
        ref = (y_prob[mask_base] >= 0.5).mean()

    for g in sorted(set(grupos)):
        mask = grupos == g
        yp_g = y_prob[mask]
        yt_g = y_true[mask]
        # Encontrar umbral que acerca la metrica al valor de referencia
        mejor_umbral = 0.5
        mejor_dif    = float("inf")
        for t in np.linspace(0.1, 0.9, 81):
            preds = (yp_g >= t).astype(int)
            if objetivo == "equal_opportunity":
                tp = ((preds == 1) & (yt_g == 1)).sum()
                fn = ((preds == 0) & (yt_g == 1)).sum()
                metrica = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                metrica = preds.mean()
            dif = abs(metrica - ref)
            if dif < mejor_dif:
                mejor_dif, mejor_umbral = dif, t
        umbrales[g] = mejor_umbral
        print(f"  Grupo '{g}': umbral optimo = {mejor_umbral:.2f}")

    return umbrales


# Aplicar umbrales calibrados
def predecir_con_umbrales(y_prob, grupos, umbrales):
    y_pred = np.zeros(len(y_prob), dtype=int)
    for g, umbral in umbrales.items():
        mask = grupos == g
        y_pred[mask] = (y_prob[mask] >= umbral).astype(int)
    return y_pred
```

---

## 4. Privacidad y proteccion de datos

### Principios basicos

- **Minimizacion de datos**: solo recolectar lo necesario para el proposito declarado.
- **Anonimizacion genuina**: eliminar no solo el nombre, sino cualquier combinacion de atributos que permita reidentificar (edad + ciudad + profesion pueden ser suficientes para identificar a alguien en datasets pequenos).
- **Privacidad diferencial**: tecnica matematica que garantiza que el resultado de un analisis no revela informacion sobre ningun individuo especifico.

### k-Anonimidad: un primer paso

Un dataset es **k-anonimo** si cada combinacion de atributos cuasi-identificadores aparece en al menos k filas. Protege contra reidentificacion pero no contra ataques de homogeneidad.

```python
def verificar_k_anonimato(df, quasi_ids, k=5):
    """
    Verifica si el dataset cumple k-anonimato.

    quasi_ids: columnas que podrian usarse para reidentificar
               (ej: ['edad_grupo', 'ciudad', 'profesion'])
    """
    grupos = df.groupby(quasi_ids).size().reset_index(name="count")
    violaciones = grupos[grupos["count"] < k]

    total_grupos    = len(grupos)
    grupos_ok       = (grupos["count"] >= k).sum()
    filas_en_riesgo = violaciones["count"].sum()

    print(f"k-Anonimato (k={k}):")
    print(f"  Total de grupos unicos: {total_grupos}")
    print(f"  Grupos con k o mas filas: {grupos_ok}")
    print(f"  Grupos con < k filas (violaciones): {len(violaciones)}")
    print(f"  Filas en riesgo de reidentificacion: {filas_en_riesgo}")

    if len(violaciones) > 0:
        print("\n  Grupos que violan k-anonimato:")
        print(violaciones.sort_values("count").head(10).to_string(index=False))
        print("\n  Mitigaciones: generalizar valores (ej: edad → rango), suprimir filas raras")
    else:
        print(f"\n  [OK] El dataset cumple {k}-anonimato")

    return violaciones


# Ejemplo de generalizacion para mejorar k-anonimato
def generalizar_edad(df, col="edad"):
    """Convierte edad exacta en grupo de 10 anos."""
    return pd.cut(df[col], bins=[0, 20, 30, 40, 50, 60, 70, 120],
                  labels=["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"])


# Privacidad diferencial con Microsoft SmartNoise / Google's DP library
# pip install diffprivlib
def estadisticas_con_dp(datos, epsilon=1.0):
    """
    Calcula media y desviacion estandar con privacidad diferencial.
    epsilon: presupuesto de privacidad (menor = mas privado, menos preciso)
    """
    import diffprivlib
    mean_dp = diffprivlib.tools.mean(datos, epsilon=epsilon,
                                     bounds=(datos.min(), datos.max()))
    std_dp  = diffprivlib.tools.std(datos, epsilon=epsilon,
                                    bounds=(datos.min(), datos.max()))
    print(f"  Media real: {datos.mean():.4f}  |  Media DP (ε={epsilon}): {mean_dp:.4f}")
    print(f"  Std real:   {datos.std():.4f}   |  Std DP  (ε={epsilon}): {std_dp:.4f}")
    return mean_dp, std_dp
```

---

## 5. Explicabilidad e interpretabilidad

Un modelo que no puede explicar sus decisiones no puede auditarse ni corregirse. En contextos de alto impacto (credito, salud, justicia), la explicabilidad es un requisito legal en muchos paises (GDPR en Europa).

```python
# ── SHAP: explicaciones locales y globales para cualquier modelo ──────────────
# pip install shap

import shap

def explicar_modelo(modelo, X_train, X_test, feature_names=None, tipo="tree"):
    """
    Genera explicaciones SHAP para un modelo.

    tipo: "tree"   -> TreeExplainer (XGBoost, LGBM, RandomForest)
          "linear" -> LinearExplainer (Logistic Regression, Linear SVM)
          "kernel" -> KernelExplainer (cualquier modelo, mas lento)
    """
    if feature_names is not None and hasattr(X_train, 'values'):
        X_train_arr = X_train.values
        X_test_arr  = X_test.values
    else:
        X_train_arr = X_train
        X_test_arr  = X_test

    if tipo == "tree":
        explainer   = shap.TreeExplainer(modelo)
    elif tipo == "linear":
        explainer   = shap.LinearExplainer(modelo, X_train_arr)
    else:
        explainer   = shap.KernelExplainer(modelo.predict_proba,
                                           shap.sample(X_train_arr, 100))

    shap_values = explainer.shap_values(X_test_arr)

    # Si clasificacion binaria: shap_values puede ser lista [clase_0, clase_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # explicaciones para clase positiva

    # Importancia global: media del valor absoluto de SHAP por feature
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X_test_arr.shape[1])]

    importancia_global = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)

    print("Top 10 features por importancia SHAP global:")
    print(importancia_global.head(10))

    # Explicacion local: una prediccion especifica
    idx = 0
    print(f"\nExplicacion local para la muestra {idx}:")
    contribuciones = pd.Series(shap_values[idx], index=feature_names)
    print(contribuciones.abs().sort_values(ascending=False).head(5))

    return shap_values, importancia_global


# ── LIME: explicaciones locales modelo-agnosticas ────────────────────────────
# pip install lime
# from lime.lime_tabular import LimeTabularExplainer
#
# explainer = LimeTabularExplainer(
#     X_train, feature_names=feature_names,
#     class_names=["negativo", "positivo"], mode="classification"
# )
# exp = explainer.explain_instance(X_test[0], modelo.predict_proba, num_features=5)
# exp.show_in_notebook()
```

---

## 6. Alucinaciones en modelos generativos

Los LLMs pueden generar texto que parece factual pero es incorrecto. Este fenomeno — llamado "alucinacion" — tiene tres variantes:

| Tipo                    | Descripcion                                          | Ejemplo                                             |
| ----------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| **Confabulacion**       | El modelo inventa hechos coherentes                  | Citar un paper que no existe con DOI inventado      |
| **Alucinacion factual** | Afirmaciones incorrectas sobre hechos reales         | Decir que el presidente de un pais es X cuando es Y |
| **Inconsistencia**      | El modelo se contradice dentro de la misma respuesta | Afirmar A en el parrafo 1 y no-A en el parrafo 3    |

### Como mitigar alucinaciones en produccion

```python
# ── Retrieval-Augmented Generation (RAG): fundamentar respuestas en fuentes ───
# La idea: antes de generar, recuperar documentos relevantes del corpus
# y pasarlos como contexto al modelo. El modelo solo debe "resumir" informacion
# que tiene delante — no inventar desde parametros.

# Pipeline conceptual de RAG:
# 1. Indexar documentos confiables (embeddings en vector store)
# 2. Query del usuario -> recuperar top-K documentos similares
# 3. Concatenar documentos + query -> enviar al LLM
# 4. El LLM genera la respuesta citando los documentos

# Con LangChain + FAISS (ejemplo simplificado):
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFacePipeline
#
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# vectorstore = FAISS.from_documents(documentos, embeddings)
# retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
# chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,
#                                      return_source_documents=True)
# respuesta = chain({"query": pregunta_usuario})


# ── Tecnicas de verificacion de outputs generativos ──────────────────────────
def validar_output_llm(texto_generado, hechos_esperados, umbral_sim=0.7):
    """
    Verifica que el texto generado mencione los hechos esperados.
    Usa embeddings de oraciones para busqueda semantica.

    hechos_esperados: lista de strings que el output debe mencionar
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb_output = model.encode([texto_generado])
    emb_hechos = model.encode(hechos_esperados)

    sims = cosine_similarity(emb_output, emb_hechos)[0]
    resultados = []
    for hecho, sim in zip(hechos_esperados, sims):
        estado = "presente" if sim >= umbral_sim else "AUSENTE/INCORRECTO"
        resultados.append({"hecho": hecho, "similitud": sim, "estado": estado})
        print(f"  [{estado}] sim={sim:.3f} | {hecho[:60]}")
    return resultados
```

---

## 7. Ficha de modelo (Model Card)

Una **Model Card** es la documentacion minima que debe acompanar a cualquier modelo en produccion o publicado. El formato fue propuesto por Mitchell et al. (Google, 2019) y es el estandar de facto en Hugging Face.

```markdown
# Model Card: Clasificador de Riesgo de Credito v1.2

## Descripcion

Modelo de clasificacion binaria que predice si un solicitante
de credito sera aprobado (1) o rechazado (0).

## Uso previsto

- **Uso recomendado**: apoyo a analistas humanos en evaluacion inicial
- **Uso NO recomendado**: decision automatica sin revision humana;
  aplicacion en poblaciones sin representacion en el dataset de entrenamiento

## Datos de entrenamiento

- Fuente: registros internos (2018-2023), 85.000 solicitudes
- Periodo cubierto: enero 2018 — diciembre 2023
- Exclusiones: menores de 18 anos, solicitudes fuera del pais X

## Metricas globales

| Metrica  | Valor |
| -------- | ----- |
| F1-macro | 0.847 |
| ROC-AUC  | 0.912 |
| Accuracy | 0.881 |

## Metricas por subgrupo (auditoria de equidad)

| Grupo              | F1    | FPR   | FNR   |
| ------------------ | ----- | ----- | ----- |
| Hombres (n=52k)    | 0.851 | 0.082 | 0.119 |
| Mujeres (n=33k)    | 0.839 | 0.094 | 0.128 |
| 18-30 anos (n=28k) | 0.821 | 0.108 | 0.143 |
| 31-50 anos (n=42k) | 0.856 | 0.076 | 0.112 |
| 51+ anos (n=15k)   | 0.831 | 0.091 | 0.132 |

## Sesgos y limitaciones conocidas

- Rendimiento inferior en solicitantes de 18-30 anos (subrepresentados en datos)
- No evaluar en regiones con distribuciones demograficas muy distintas al dataset
- El modelo no tiene en cuenta contexto economico macro (inflacion, desempleo)

## Consideraciones eticas

- Decisiones de credito son de alto impacto — siempre supervisar con analistas
- Monitorear drift mensualmente y reentrenar si F1 cae > 3 puntos
- Auditar fairness cada 6 meses con datos recientes

## Informacion de contacto y versionado

- Entrenado por: Equipo de Riesgo, Empresa X
- Fecha de ultima actualizacion: 2024-03-15
- Version del modelo: 1.2
- Repositorio: [enlace interno]
```

```python
# Automatizar la generacion de metricas para la model card

def generar_metricas_model_card(modelo, X_test, y_test, grupos_dict, feature_names=None):
    """
    genera_metricas_model_card: calcula y formatea metricas para una Model Card.

    grupos_dict: dict {"nombre_grupo": array_de_grupos}
    """
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    print("## Metricas Globales")
    print(f"| F1-macro  | {f1_score(y_test, y_pred, average='macro'):.4f} |")
    print(f"| Accuracy  | {accuracy_score(y_test, y_pred):.4f} |")
    if y_prob is not None:
        print(f"| ROC-AUC   | {roc_auc_score(y_test, y_prob):.4f} |")

    print("\n## Metricas por Subgrupo")
    print(f"| Grupo | F1 | FPR | FNR | n |")
    print(f"|-------|-----|-----|-----|---|")
    for nombre, grupos in grupos_dict.items():
        for g in sorted(set(grupos)):
            mask = grupos == g
            yt, yp = y_test[mask], y_pred[mask]
            f1  = f1_score(yt, yp, average="macro")
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
            print(f"| {nombre}={g} | {f1:.3f} | {fpr:.3f} | {fnr:.3f} | {mask.sum()} |")
```

---

## 8. Checklist de despliegue responsable

Antes de poner un modelo en produccion o publicar en una competencia, verifica:

**Datos:**

- [ ] El dataset tiene documentacion de origen y fecha de corte
- [ ] Se verifico que no haya data leakage
- [ ] Se evaluo representatividad de subgrupos criticos
- [ ] Los datos sensibles estan protegidos o anonimizados

**Modelo:**

- [ ] Se mide rendimiento por subgrupos (no solo global)
- [ ] Se identificaron los casos de uso fuera de distribucion
- [ ] El modelo tiene umbrales documentados y justificados
- [ ] Se documento la model card con limitaciones conocidas

**Despliegue:**

- [ ] Hay monitoreo de drift en produccion
- [ ] Existe un mecanismo de reentrenamiento periodico
- [ ] Hay un proceso de revision humana para casos de alto impacto
- [ ] Se puede revertir el modelo rapidamente si hay problemas

**Comunicacion:**

- [ ] Los usuarios finales entienden que el modelo puede equivocarse
- [ ] Los limites del modelo estan documentados en lenguaje claro
- [ ] Hay un canal para reportar fallas o danos detectados

---

## Recursos recomendados

- [**Principios de IA de la OCDE**](https://oecd.ai/en/ai-principles): marco internacional de referencia para el uso responsable de IA, adoptado por mas de 40 paises
- [**UNESCO — Recomendacion sobre Etica de la IA**](https://www.unesco.org/en/artificial-intelligence/recommendation-ethics): primer instrumento normativo global, con guia de implementacion
- [**IBM AI Fairness 360**](https://aif360.mybluemix.net/): toolkit open-source con 70+ metricas de fairness y algoritmos de mitigacion
- [**Fairness and Machine Learning (libro gratuito)**](https://fairmlbook.org/): fundamentos matematicos de justicia algoritmica con ejemplos reales
- [**Model Cards for Model Reporting (Mitchell et al.)**](https://arxiv.org/abs/1810.03993): el paper original que propone el formato de model card

---

## Navegacion

[← 16. Flujo de Trabajo en Kaggle y Competencias](/ruta-aprendizaje/16-flujo-de-trabajo-en-kaggle-y-competencias) | [18. Series Temporales y Datos Secuenciales →](/ruta-aprendizaje/18-series-temporales-y-datos-secuenciales)
