# FashRec

## El problema

Un catálogo de moda de Amazon tiene cientos de miles de productos y millones de reseñas, pero cada usuario solo ha visto una fracción mínima. FashRec resuelve la recomendación **Top-N**: dado un usuario (o su historial reciente de productos), devolver los artículos que es más probable que le interesen.

Se entrena sobre **Amazon Reviews 2023, categoría Amazon Fashion** (2 500 939 reseñas y 826 108 productos). El enfoque es un modelo **Two-Tower** (una torre para el usuario, otra para el producto) que proyecta ambos a un mismo espacio vectorial; las recomendaciones se obtienen por similitud con búsqueda **FAISS**. Se compara contra dos líneas base (popularidad e ItemKNN) con un protocolo de corte temporal. En el modelo publicado, el Two-Tower supera a popularidad en test (Recall@20 ≈ 0.077 frente a ≈ 0.057).

El flujo de datos es batch:

```
JSONL crudos → PySpark (ETL) → Parquet → features + protocolo temporal
→ Two-Tower (PyTorch) → FAISS → recomendaciones precalculadas → Streamlit
```

El pipeline corre igual en local (un directorio `data/`) que en AWS (el mismo árbol bajo un bucket de S3). El dashboard solo consulta resultados ya calculados.

---

## Fuentes de datos

- **Dataset:** Amazon Reviews 2023 — McAuley Lab, UC San Diego. https://amazon-reviews-2023.github.io/
- **Categoría:** Amazon Fashion. Se usan dos archivos JSONL:

  | Archivo | Contenido | Tamaño | Líneas |
  |---|---|---|---|
  | `Amazon_Fashion.jsonl` | reseñas (`review` por categoría) | ≈ 1.0 GB | 2 500 939 |
  | `meta_Amazon_Fashion.jsonl` | metadatos de producto (`meta` por categoría) | ≈ 1.4 GB | 826 108 |

- **Descarga:** desde la página del dataset, sección *Grouped by Category* → *Amazon Fashion* (archivos `review` y `meta`). Descomprime los `.jsonl.gz` y coloca los dos `.jsonl` en la **raíz del repositorio** con los nombres de la tabla.
- **Integridad:** no se modifican. `data/reports/raw_manifest.json` guarda su tamaño y SHA-256; `publish_job` verifica que no hayan cambiado. Valores de referencia:
  - `Amazon_Fashion.jsonl` — SHA-256 `e597b3ce…986c57`
  - `meta_Amazon_Fashion.jsonl` — SHA-256 `629dde62…51a62df`

Los `.jsonl` y la carpeta `data/` no se versionan en git (ver `.gitignore`): se regeneran con el pipeline.

---

## Requisitos

- **Python 3.11+**
- **Java 17+** (lo necesita PySpark para el ETL local)
- Dependencias en `requirements.txt` (pipeline completo) y `requirements-app.txt` (solo el dashboard).
- Para el despliegue en AWS: WSL con Docker, AWS CLI v2 y el Serverless Framework

---

## Reproducir en local

```powershell
# 1. Entorno
python -m venv .venv
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install -r requirements.txt

# 2. Pipeline completo: ETL -> features -> train -> evaluate -> recommend -> publish
.\.venv\Scripts\python -m jobs.run_pipeline

# 3. Dashboard (lee los resultados ya calculados)
.\.venv\Scripts\python -m streamlit run app/streamlit_app.py

```

**Jobs individuales** (`python -m jobs.<nombre> --help`): `etl_job`, `features_job`, `train_job`, `evaluate_job`, `recommend_job`, `publish_job`, `sweep` (ajuste de hiperparámetros, solo validación).

**Tiempos de referencia** (12 núcleos CPU, 32 GB, sin GPU): ETL ≈ 89 s · features ≈ 70–90 s · entrenamiento ≈ 55 s/época · evaluación ≈ 30 s · recomendaciones ≈ 3.5 min.

---

## Estructura del repositorio

```
Amazon_Fashion.jsonl, meta_Amazon_Fashion.jsonl   fuentes crudas (no versionadas; se descargan)
configs/default.yaml      configuración versionada del pipeline (todas las etapas)
src/                      código: data/ (ETL, features), models/ (Two-Tower), retrieval/ (FAISS),
                          evaluation/, serving/ (acceso a datos del dashboard), common/
jobs/                     un job por etapa + run_pipeline.py (orquestación local)
app/                      dashboard Streamlit (5 páginas)
data/                     data lake local (réplica del árbol de S3); ver abajo
deploy/                   despliegue en AWS (Serverless Framework); ver abajo
scripts/                  utilidades (p. ej. md_to_docx.py)
requirements.txt          dependencias del pipeline;  requirements-app.txt: solo dashboard
```

**Layout de `data/`** (idéntico en local y en S3):

```
curated/{reviews,products,interactions}/          Parquet limpio (ETL)
model_data/dataset_version=v1/                     features, mappings, splits del protocolo
artifacts/model_version=<v>/                       model.pt, faiss.index, embeddings, config.json, training_history.csv
recommendations/model_version=<v>/                 Top-N precalculado por usuario
evaluation/model_version=<v>/                       metrics.json, evaluation_details.parquet, comparison_table.csv
published/current.json                             versión activa (puntero para rollback)
runs/                                              log JSON de cada ejecución de job
reports/                                           raw_manifest.json y controles de calidad del ETL
```

---

## Reproducir en AWS

Despliegue serverless completo (S3 + Glue + Athena + DynamoDB + Fargate + Step Functions + Lambda/API Gateway + dashboard en ECS) desde WSL. Guía detallada y justificación de cada servicio en `DEPLOYMENT_SERVERLESS.md` y `DECISIONES_AWS.md`.

```bash
nano .env                       # credenciales de AWS y opciones (plantilla: .env.example)

bash deploy/deploy.sh --dry-run # muestra el plan sin tocar AWS
bash deploy/deploy.sh           # despliega según .env

bash deploy/destroy.sh --include-data   # tumba todo cuando termines (evita sobrecostos)
```
