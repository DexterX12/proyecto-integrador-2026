"""FashRec - página de inicio. Las demás vistas están en app/pages/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.components.common import get_repo, sidebar_version  # noqa: E402

st.set_page_config(page_title="FashRec", page_icon="👗", layout="wide")
repo = get_repo()
version = sidebar_version(repo)

st.title("👗 FashRec")
st.caption("Sistema de recomendación Top-N para Amazon Fashion (Amazon Reviews 2023) · Two-Tower híbrido + FAISS")

if version is None:
    st.stop()

pub = repo.published()
cfg = repo.model_config(version)
meta = repo.recommendation_meta(version)
left, right = st.columns([3, 2])
with left:
    st.markdown(
        """
**Qué hace esta aplicación**

Este dashboard **solo consulta resultados ya calculados** por el pipeline batch (no entrena ni reconstruye
índices al abrirse). Use el menú lateral:

1. **Overview**: datos curados, distribuciones y KPIs del modelo activo.
2. **Explorador de usuarios**: historial de un usuario y sus recomendaciones Top-N, con imágenes.
3. **Comparación de modelos**: Recall@K, NDCG@K, cobertura, diversidad y novedad frente a las líneas base.
4. **Explorador de productos**: búsqueda, productos cercanos (FAISS) y proyección UMAP de embeddings.
5. **Estado del pipeline**: ejecuciones, versiones publicadas y chequeos de calidad.
"""
    )
with right:
    st.subheader("Modelo activo")
    st.write(f"**Versión seleccionada:** `{version}`")
    if pub:
        st.write(f"**Versión publicada:** `{pub['model_version']}` · publicada {pub['published_at'][:16].replace('T', ' ')} UTC")
    else:
        st.warning("Aún no hay una versión publicada (jobs.publish_job).")
    st.write(f"**Recomendaciones generadas:** {meta.get('generated_at', '?')[:16].replace('T', ' ')} UTC")
    st.write(f"**Usuarios con recomendaciones:** {meta.get('users', 0):,} · Top-{meta.get('top_n', '?')}")
    st.write(f"**Historial hasta:** {meta.get('as_of', '?')}")
    hp = cfg.get("hyperparameters", {})
    if hp:
        with st.expander("Hiperparámetros"):
            st.json(hp)
