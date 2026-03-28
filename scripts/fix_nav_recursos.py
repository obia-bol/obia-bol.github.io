"""
Actualiza todos los archivos de la ruta de aprendizaje (00-20):
- Reemplaza secciones de recursos con hiperlinks reales
- Reemplaza "## Siguiente tema" con "## Navegacion" bidireccional
"""

import re
from pathlib import Path

BASE = Path("src/content/posts")

# ── Tabla de navegacion ──────────────────────────────────────────────────────
# (archivo, titulo_prev, slug_prev, titulo_next, slug_next)
NAV = {
    "00-introduccion-olimpiadas-ia.md": (
        None,
        None,
        "1. Fundamentos de Programacion en Python",
        "1-fundamentos-de-programacion-en-python",
    ),
    "01-fundamentos-python.md": (
        "0. Introduccion a las Olimpiadas de IA",
        "0-introduccion-a-las-olimpiadas-de-ia",
        "2. Matematicas para Machine Learning",
        "2-matematicas-para-machine-learning",
    ),
    "02-matematicas-ml.md": (
        "1. Fundamentos de Programacion en Python",
        "1-fundamentos-de-programacion-en-python",
        "3. Manejo de Datos con NumPy y Pandas",
        "3-manejo-de-datos-con-numpy-y-pandas",
    ),
    "03-numpy-pandas.md": (
        "2. Matematicas para Machine Learning",
        "2-matematicas-para-machine-learning",
        "4. Visualizacion de Datos",
        "4-visualizacion-de-datos",
    ),
    "04-visualizacion-datos.md": (
        "3. Manejo de Datos con NumPy y Pandas",
        "3-manejo-de-datos-con-numpy-y-pandas",
        "5. Introduccion a Machine Learning",
        "5-introduccion-a-machine-learning",
    ),
    "05-introduccion-ml.md": (
        "4. Visualizacion de Datos",
        "4-visualizacion-de-datos",
        "6. Fundamentos de Scikit-Learn",
        "6-fundamentos-de-scikit-learn",
    ),
    "06-scikit-learn.md": (
        "5. Introduccion a Machine Learning",
        "5-introduccion-a-machine-learning",
        "7. Modelos de Regresion",
        "7-modelos-de-regresion",
    ),
    "07-modelos-regresion.md": (
        "6. Fundamentos de Scikit-Learn",
        "6-fundamentos-de-scikit-learn",
        "8. Modelos de Clasificacion",
        "8-modelos-de-clasificacion",
    ),
    "08-modelos-clasificacion.md": (
        "7. Modelos de Regresion",
        "7-modelos-de-regresion",
        "9. Aprendizaje No Supervisado",
        "9-aprendizaje-no-supervisado",
    ),
    "09-aprendizaje-no-supervisado.md": (
        "8. Modelos de Clasificacion",
        "8-modelos-de-clasificacion",
        "10. Introduccion a Redes Neuronales",
        "10-introduccion-a-redes-neuronales",
    ),
    "10-intro-redes-neuronales.md": (
        "9. Aprendizaje No Supervisado",
        "9-aprendizaje-no-supervisado",
        "11. Fundamentos de PyTorch",
        "11-fundamentos-de-pytorch",
    ),
    "11-pytorch-fundamentos.md": (
        "10. Introduccion a Redes Neuronales",
        "10-introduccion-a-redes-neuronales",
        "12. Tecnicas de Entrenamiento en Deep Learning",
        "12-tecnicas-de-entrenamiento-en-deep-learning",
    ),
    "12-tecnicas-entrenamiento-dl.md": (
        "11. Fundamentos de PyTorch",
        "11-fundamentos-de-pytorch",
        "13. Redes Convolucionales (CNNs)",
        "13-redes-convolucionales-cnns",
    ),
    "13-cnns.md": (
        "12. Tecnicas de Entrenamiento en Deep Learning",
        "12-tecnicas-de-entrenamiento-en-deep-learning",
        "14. Fundamentos de NLP",
        "14-fundamentos-de-nlp",
    ),
    "14-fundamentos-nlp.md": (
        "13. Redes Convolucionales (CNNs)",
        "13-redes-convolucionales-cnns",
        "15. Embeddings y Transformers",
        "15-embeddings-y-transformers",
    ),
    "15-embeddings-transformers.md": (
        "14. Fundamentos de NLP",
        "14-fundamentos-de-nlp",
        "16. Flujo de Trabajo en Kaggle y Competencias",
        "16-flujo-de-trabajo-en-kaggle-y-competencias",
    ),
    "16-flujo-competencias-kaggle.md": (
        "15. Embeddings y Transformers",
        "15-embeddings-y-transformers",
        "17. Etica y IA Responsable",
        "17-etica-y-ia-responsable",
    ),
    "17-etica-ia-responsable.md": (
        "16. Flujo de Trabajo en Kaggle y Competencias",
        "16-flujo-de-trabajo-en-kaggle-y-competencias",
        "18. Prompt Engineering y Fundamentos de LLMs",
        "18-prompt-engineering-y-fundamentos-de-llms",
    ),
    "18-prompt-engineering-llm.md": (
        "17. Etica y IA Responsable",
        "17-etica-y-ia-responsable",
        "19. Series Temporales y Datos Secuenciales",
        "19-series-temporales-y-datos-secuenciales",
    ),
    "19-series-temporales.md": (
        "18. Prompt Engineering y Fundamentos de LLMs",
        "18-prompt-engineering-y-fundamentos-de-llms",
        "20. Proyectos Finales y Mock Competitions",
        "20-proyectos-finales-y-mock-competitions",
    ),
    "20-proyectos-finales.md": (
        "19. Series Temporales y Datos Secuenciales",
        "19-series-temporales-y-datos-secuenciales",
        None,
        None,
    ),
}

# ── Secciones de recursos por archivo ────────────────────────────────────────
RECURSOS = {
    "00-introduccion-olimpiadas-ia.md": """\
## Recursos recomendados

- [**Sitio oficial IOAI**](https://ioai-official.org/): noticias, syllabus, reglas y problemas anteriores de la olimpiada
- [**Kaggle Learn — Intro to Machine Learning**](https://www.kaggle.com/learn/intro-to-machine-learning): introduccion practica gratuita sin prerequisitos
- [**Google PAIR — Explorables de IA**](https://pair.withgoogle.com/explorables/): visualizaciones interactivas sobre conceptos de ML y etica
- [**IBM AI Fairness 360**](https://aif360.mybluemix.net/): documentacion sobre fairness y sesgo en IA""",
    "01-fundamentos-python.md": """\
## Recursos recomendados

- [**Documentacion oficial de Python**](https://docs.python.org/3/tutorial/): tutorial completo del lenguaje, disponible en espanol
- [**Real Python**](https://realpython.com/): articulos y tutoriales sobre fundamentos, OOP y buenas practicas
- [**Kaggle Learn — Python**](https://www.kaggle.com/learn/python): curso interactivo gratuito con notebooks ejecutables
- [**Kaggle Learn — Intro to Machine Learning**](https://www.kaggle.com/learn/intro-to-machine-learning): siguiente paso natural despues de los fundamentos""",
    "02-matematicas-ml.md": """\
## Recursos recomendados

- [**3Blue1Brown — Algebra Lineal**](https://www.3blue1brown.com/topics/linear-algebra): la mejor visualizacion geometrica de vectores, matrices y transformaciones
- [**3Blue1Brown — Redes Neuronales**](https://www.3blue1brown.com/topics/neural-networks): intuicion sobre gradientes y backpropagation con animaciones
- [**Khan Academy — Calculo**](https://www.khanacademy.org/math/calculus-1): derivadas e integrales desde cero, completamente gratuito
- [**Khan Academy — Estadistica y Probabilidad**](https://www.khanacademy.org/math/statistics-probability): distribuciones, inferencia y probabilidad condicional
- [**Mathematics for Machine Learning (Coursera)**](https://www.coursera.org/specializations/mathematics-machine-learning): especialidad completa que conecta matematicas con ML""",
    "03-numpy-pandas.md": """\
## Recursos recomendados

- [**Documentacion oficial de NumPy**](https://numpy.org/doc/stable/): referencia completa de arrays, operaciones y broadcasting
- [**Documentacion oficial de Pandas**](https://pandas.pydata.org/docs/): guia completa de DataFrames, IO y transformaciones
- [**Kaggle Learn — Pandas**](https://www.kaggle.com/learn/pandas): curso interactivo gratuito con ejercicios reales
- [**Kaggle Learn — Data Cleaning**](https://www.kaggle.com/learn/data-cleaning): tecnicas de limpieza aplicadas a datasets reales
- [**Kaggle — Dataset Titanic**](https://www.kaggle.com/competitions/titanic): dataset clasico para practicar EDA y manipulacion de datos""",
    "04-visualizacion-datos.md": """\
## Recursos recomendados

- [**Documentacion de Matplotlib**](https://matplotlib.org/stable/): referencia completa con galeria de ejemplos
- [**Documentacion de Seaborn**](https://seaborn.pydata.org/): visualizaciones estadisticas con menos codigo
- [**Kaggle Learn — Data Visualization**](https://www.kaggle.com/learn/data-visualization): curso interactivo con Seaborn
- [**From Data to Viz**](https://www.data-to-viz.com/): guia de decision sobre que grafico usar segun tipo de datos
- [**Kaggle — House Prices**](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques): excelente dataset para practicar EDA y visualizacion""",
    "05-introduccion-ml.md": """\
## Recursos recomendados

- [**Hands-On Machine Learning (Geron, O'Reilly)**](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/): el libro practico mas completo de ML con sklearn y Keras
- [**Kaggle Learn — Intro to Machine Learning**](https://www.kaggle.com/learn/intro-to-machine-learning): introduccion practica gratuita
- [**Kaggle Learn — Intermediate Machine Learning**](https://www.kaggle.com/learn/intermediate-machine-learning): validacion, pipelines y manejo de datos faltantes
- [**Machine Learning Specialization (Andrew Ng, Coursera)**](https://www.coursera.org/specializations/machine-learning-introduction): el curso de ML mas conocido del mundo""",
    "06-scikit-learn.md": """\
## Recursos recomendados

- [**Documentacion oficial de scikit-learn**](https://scikit-learn.org/stable/): referencia completa con ejemplos para cada modelo y utilidad
- [**User Guide — Model Selection**](https://scikit-learn.org/stable/model_selection.html): validacion cruzada, metricas y busqueda de hiperparametros
- [**User Guide — Pipelines**](https://scikit-learn.org/stable/modules/compose.html): construccion de pipelines reproducibles y sin leakage
- [**Kaggle Learn — Intro to Machine Learning**](https://www.kaggle.com/learn/intro-to-machine-learning): practica con sklearn desde cero
- [**Kaggle Learn — Intermediate Machine Learning**](https://www.kaggle.com/learn/intermediate-machine-learning): pipelines y validacion avanzada""",
    "07-modelos-regresion.md": """\
## Recursos recomendados

- [**Documentacion sklearn: modelos lineales**](https://scikit-learn.org/stable/modules/linear_model.html): referencia completa de regresion lineal, Ridge, Lasso y ElasticNet
- [**ISLR capitulo 3 — Linear Regression**](https://www.statlearning.com/) (libro gratuito): fundamentos estadisticos de regresion
- [**ISLR capitulo 6 — Regularization**](https://www.statlearning.com/) (libro gratuito): Ridge, Lasso y seleccion de variables
- [**Visualizacion interactiva de regularizacion**](https://explained.ai/regularization/): intuicion geometrica sobre Ridge y Lasso""",
    "08-modelos-clasificacion.md": """\
## Recursos recomendados

- [**Documentacion sklearn: clasificadores**](https://scikit-learn.org/stable/supervised_learning.html): referencia de todos los clasificadores con ejemplos
- [**XGBoost docs**](https://xgboost.readthedocs.io/): documentacion oficial con guias de uso y parametros
- [**ISLR capitulos 4 y 8**](https://www.statlearning.com/) (libro gratuito): clasificacion y metodos basados en arboles
- [**imbalanced-learn**](https://imbalanced-learn.org/stable/): guia practica de metricas y tecnicas para datos desbalanceados""",
    "09-aprendizaje-no-supervisado.md": """\
## Recursos recomendados

- [**Documentacion sklearn: clustering**](https://scikit-learn.org/stable/modules/clustering.html)
- [**Documentacion sklearn: PCA y descomposicion**](https://scikit-learn.org/stable/modules/decomposition.html)
- [**ISLR capitulo 12 — Unsupervised Learning**](https://www.statlearning.com/) (libro gratuito)
- [**Tutorial t-SNE por Laurens van der Maaten**](https://lvdmaaten.github.io/tsne/)
- [**UMAP: Uniform Manifold Approximation**](https://umap-learn.readthedocs.io/)""",
    "10-intro-redes-neuronales.md": None,  # ya tiene recursos correctos
    "11-pytorch-fundamentos.md": None,  # ya tiene recursos correctos
    "12-tecnicas-entrenamiento-dl.md": """\
## Recursos recomendados

- [**CS231n — Notas de entrenamiento**](https://cs231n.github.io/neural-networks-3/): guia detallada sobre optimizacion, learning rate y diagnostico
- [**Documentacion de optimizadores en PyTorch**](https://pytorch.org/docs/stable/optim.html): referencia completa de SGD, Adam, AdamW y schedulers
- [**Batch Normalization (paper original)**](https://arxiv.org/abs/1502.03167): articulo de Ioffe y Szegedy, lectura recomendada
- [**Dropout (Srivastava et al.)**](https://jmlr.org/papers/v15/srivastava14a.html): paper original de dropout como regularizacion""",
    "13-cnns.md": """\
## Recursos recomendados

- [**CS231n — Convolutional Neural Networks**](https://cs231n.github.io/convolutional-networks/): explicacion visual completa de convoluciones, pooling y arquitecturas
- [**Tutorial de transfer learning en PyTorch**](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): guia oficial con ResNet en datos personalizados
- [**Papers With Code — Image Classification**](https://paperswithcode.com/task/image-classification): estado del arte y benchmarks en clasificacion de imagenes
- [**TorchVision**](https://pytorch.org/vision/stable/): modelos preentrenados, datasets y transformaciones de imagen""",
    "14-fundamentos-nlp.md": """\
## Recursos recomendados

- [**Documentacion de NLTK**](https://www.nltk.org/): libreria clasica de NLP con tokenizacion, stemming y parsers
- [**Documentacion de spaCy**](https://spacy.io/usage): NLP industrial, rapido y con modelos preentrenados en espanol
- [**Hugging Face NLP Course — Modulos 1-3**](https://huggingface.co/learn/nlp-course): introduccion moderna al NLP con transformers
- [**Kaggle Learn — Natural Language Processing**](https://www.kaggle.com/learn/natural-language-processing): practica rapida con TF-IDF y embeddings""",
    "15-embeddings-transformers.md": """\
## Recursos recomendados

- [**The Illustrated Transformer (Jay Alammar)**](https://jalammar.github.io/illustrated-transformer/): la explicacion visual mas clara del mecanismo de atencion
- [**Curso oficial de Hugging Face**](https://huggingface.co/learn/nlp-course): desde tokenizacion hasta fine-tuning de BERT y GPT
- [**Documentacion de `transformers`**](https://huggingface.co/docs/transformers): referencia completa de la libreria con ejemplos
- [**Word2Vec (Mikolov et al.)**](https://arxiv.org/abs/1301.3781): el paper original de embeddings de palabras
- [**BERT (Devlin et al.)**](https://arxiv.org/abs/1810.04805): fundamento de los modelos modernos de lenguaje""",
    "16-flujo-competencias-kaggle.md": """\
## Recursos recomendados

- [**Kaggle Learn — Feature Engineering**](https://www.kaggle.com/learn/feature-engineering): tecnicas practicas de transformacion de variables
- [**Kaggle — Soluciones y write-ups**](https://www.kaggle.com/discussions?category=competitions): soluciones de ganadores con decisiones tecnicas justificadas
- [**MLflow**](https://mlflow.org/): tracking de experimentos, parametros y metricas de forma reproducible
- [**Weights & Biases**](https://wandb.ai/): plataforma de seguimiento y visualizacion de experimentos""",
    "17-etica-ia-responsable.md": """\
## Recursos recomendados

- [**Principios de IA de la OCDE**](https://oecd.ai/en/ai-principles): marco internacional de referencia para el uso responsable de IA
- [**UNESCO — Recomendacion sobre Etica de la IA**](https://www.unesco.org/en/artificial-intelligence/recommendation-ethics): primer instrumento normativo global de etica en IA
- [**Google Responsible AI Practices**](https://ai.google/responsibility/responsible-ai-practices/): guias practicas de diseno responsable
- [**IBM AI Fairness 360**](https://aif360.mybluemix.net/): toolkit de codigo abierto para medir y mitigar sesgo
- [**Fairness and Machine Learning (libro)**](https://fairmlbook.org/): libro gratuito sobre justicia algoritmica""",
    "18-prompt-engineering-llm.md": """\
## Recursos recomendados

- [**Prompt Engineering for Developers (DeepLearning.AI)**](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/): curso gratuito sobre prompting efectivo con LLMs
- [**Documentacion de la API de OpenAI**](https://platform.openai.com/docs): referencia de endpoints, modelos y buenas practicas
- [**Hugging Face — Text Generation**](https://huggingface.co/docs/transformers/tasks/language_modeling): uso de modelos open-source para generacion de texto
- [**LangChain**](https://python.langchain.com/docs/): framework para construir aplicaciones con LLMs y RAG
- [**Ollama**](https://ollama.com/): herramienta para correr modelos open-source localmente""",
    "19-series-temporales.md": """\
## Recursos recomendados

- [**statsmodels — Series Temporales**](https://www.statsmodels.org/stable/tsa.html): ARIMA, descomposicion y diagnostico de series temporales
- [**Tutorial de LSTM en PyTorch**](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html): implementacion paso a paso de redes recurrentes
- [**Kaggle Learn — Time Series**](https://www.kaggle.com/learn/time-series): forecasting con features de lag y modelos gradient boosting
- [**Darts**](https://unit8co.github.io/darts/): libreria moderna para forecasting con multiples modelos y metricas""",
    "20-proyectos-finales.md": """\
## Recursos recomendados

- [**Kaggle — Competitions**](https://www.kaggle.com/competitions): plataforma para participar en competencias reales con datasets y leaderboard
- [**MLflow**](https://mlflow.org/): herramienta de tracking de experimentos y gestion de modelos
- [**Weights & Biases**](https://wandb.ai/): visualizacion y comparacion de experimentos en tiempo real
- [**Cookiecutter Data Science**](https://drivendata.github.io/cookiecutter-data-science/): estructura estandar de proyecto reproducible""",
}


def build_nav(fname):
    pt, ps, nt, ns = NAV[fname]
    parts = []
    if ps:
        parts.append(f"[← {pt}](/ruta-aprendizaje/{ps})")
    if ns:
        parts.append(f"[{nt} →](/ruta-aprendizaje/{ns})")
    return "## Navegacion\n\n" + " | ".join(parts)


def update_file(fname):
    path = BASE / fname
    if not path.exists():
        print(f"  SKIP (no existe): {fname}")
        return

    text = path.read_text(encoding="utf-8")
    original = text

    # 1. Reemplazar contenido de "## Recursos recomendados"
    new_rec = RECURSOS.get(fname)
    if new_rec:
        # Captura desde "## Recursos recomendados" hasta el proximo "##" o fin
        text = re.sub(
            r"## Recursos recomendados\n[\s\S]*?(?=\n## |\Z)",
            new_rec + "\n",
            text,
        )

    # 2. Construir bloque de navegacion
    nav_block = build_nav(fname)

    # 3. Eliminar cualquier "## Siguiente tema" block (hasta fin de archivo)
    text = re.sub(r"\n## Siguiente tema\n[\s\S]*$", "", text)

    # 4. Eliminar "## Navegacion" existente si ya existe (para reinsertar limpio)
    text = re.sub(r"\n## Navegacion\n[\s\S]*?(?=\n## |\Z)", "\n", text)

    # 5. Agregar navegacion al final (despues del ultimo bloque de contenido)
    text = text.rstrip() + "\n\n---\n\n" + nav_block + "\n"

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"  OK: {fname}")
    else:
        print(f"  SIN CAMBIOS: {fname}")


print("Actualizando archivos...")
for fname in NAV:
    update_file(fname)
print("Listo.")
