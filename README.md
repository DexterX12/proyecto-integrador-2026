# FashRec

Recomendador Top-N para Amazon Fashion (Amazon Reviews 2023). Pipeline batch local según `plan-arquitectura-fashrec.md`:
**PySpark (ETL) → Parquet → features/protocolo temporal → Two-Tower (PyTorch) → FAISS → recomendaciones precalculadas → Streamlit (DuckDB)**.
El dashboard solo lee resultados ya calculados.

## Uso

```powershell
python -m venv .venv ; .\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install -r requirements.txt            # requiere Java 17+ para PySpark

.\.venv\Scripts\python -m jobs.run_pipeline                # ETL -> features -> train -> evaluate -> recommend -> publish
.\.venv\Scripts\python -m jobs.run_pipeline --from evaluate --model-version <v>   # reanudar desde una etapa
.\.venv\Scripts\python -m streamlit run app/streamlit_app.py
.\.venv\Scripts\python -m pytest
```

Jobs individuales (`python -m jobs.<nombre> --help`): `etl_job`, `features_job`, `train_job`, `evaluate_job`, `recommend_job`, `publish_job` (incluye `--rollback`), `sweep` (ajuste de hiperparámetros, solo validación).
Toda la configuración está en `configs/default.yaml`; se sobreescribe con `--set train.epochs=5 model.embedding_dim=128`.
Tiempos medidos (12 núcleos CPU, 32 GB): ETL 89 s, features 70–90 s, entrenamiento ≈ 55 s/época, evaluación 30 s, recomendaciones 3.5 min.

## Layout de datos (`data/`, réplica de S3)

`curated/{reviews,products,interactions}` · `model_data/dataset_version=v1/` · `artifacts/model_version=<v>/` (model.pt, faiss.index, embeddings, id_mappings, config.json, training_history.csv) · `recommendations/model_version=<v>/` · `evaluation/model_version=<v>/` (metrics.json, evaluation_details.parquet, comparison_table.csv) · `published/current.json` · `runs/` (log de cada job) · `reports/`.
Los JSONL originales no se modifican; `reports/raw_manifest.json` guarda su SHA-256 y `publish_job` lo verifica.