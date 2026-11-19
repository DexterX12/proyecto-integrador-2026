"""Componentes compartidos del dashboard (selector de versión, tarjetas con imagen y fallback)."""
from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.serving.repository import Repository  # noqa: E402

# Imagen de reemplazo (SVG embebido) cuando la URL falta o la carga falla.
PLACEHOLDER = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'>"
               "<rect width='100%25' height='100%25' fill='%23eceff1'/>"
               "<text x='50%25' y='50%25' fill='%2390a4ae' font-size='14' text-anchor='middle' "
               "dominant-baseline='middle' font-family='sans-serif'>sin imagen</text></svg>")


@st.cache_resource(show_spinner=False)
def get_repo() -> Repository:
    return Repository()


def sidebar_version(repo: Repository) -> str | None:
    """Selector de versión de modelo compartido entre páginas."""
    versions = repo.model_versions()
    if not versions:
        st.sidebar.error("No hay versiones de modelo con recomendaciones y evaluación. "
                         "Ejecute el pipeline (python -m jobs.run_pipeline).")
        return None
    default = repo.default_version()
    current = st.session_state.get("_model_version", default)
    if current not in versions:
        current = default
    version = st.sidebar.selectbox("Versión del modelo", versions, index=versions.index(current),
                                   help="La versión publicada se marca con (activa).",
                                   format_func=lambda v: f"{v} (activa)" if v == default else v)
    st.session_state["_model_version"] = version
    cfg = repo.model_config(version)
    st.sidebar.caption(f"Entrenado: {cfg.get('created_at', '?')[:16].replace('T', ' ')} UTC  \n"
                       f"Dataset: {cfg.get('dataset_version', '?')}  \n"
                       f"Corte de recomendaciones: {cfg.get('cutoffs', {}).get('test_start', '?')}")
    return version


def fmt_price(p) -> str:
    return "s/d" if p is None or pd.isna(p) else f"${p:,.2f}"


def fmt_date(d) -> str:
    if d is None or pd.isna(d):
        return ""
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def product_cards(rows: list[dict], columns: int = 5, height_per_row: int = 330) -> None:
    """Rejilla de tarjetas. Cada dict admite: title, image_url, subtitle, badges[(texto, color)], lines[str]."""
    if not rows:
        st.info("Sin productos para mostrar.")
        return
    cards = []
    for r in rows:
        title = html.escape((r.get("title") or "(sin título)")[:110])
        img = html.escape(r.get("image_url") or "", quote=True) or PLACEHOLDER
        badges = "".join(f"<span class='b' style='background:{c}'>{html.escape(t)}</span>"
                         for t, c in r.get("badges", []))
        lines = "".join(f"<div class='l'>{html.escape(str(x))}</div>" for x in r.get("lines", []) if x)
        cards.append(f"""
        <div class='card'>
          <div class='imgw'><img src="{img}" loading="lazy" referrerpolicy="no-referrer"
               onerror="this.onerror=null;this.src=&quot;{PLACEHOLDER}&quot;"/></div>
          <div class='bd'>{badges}</div>
          <div class='t' title="{title}">{title}</div>{lines}
        </div>""")
    n_rows = -(-len(rows) // columns)
    doc = f"""
    <style>
      body {{ margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; }}
      .grid {{ display:grid; grid-template-columns: repeat({columns}, 1fr); gap:12px; }}
      .card {{ border:1px solid #e0e0e0; border-radius:10px; padding:8px; background:#fff; overflow:hidden; }}
      .imgw {{ height:150px; display:flex; align-items:center; justify-content:center; background:#fafafa;
               border-radius:6px; overflow:hidden; }}
      .imgw img {{ max-width:100%; max-height:150px; object-fit:contain; }}
      .t {{ font-size:12.5px; font-weight:600; margin:6px 0 2px; line-height:1.25; height:48px; overflow:hidden; color:#263238; }}
      .l {{ font-size:11.5px; color:#546e7a; line-height:1.35; }}
      .bd {{ margin-top:6px; min-height:18px; }}
      .b {{ color:#fff; font-size:10.5px; padding:1px 7px; border-radius:9px; margin-right:4px; display:inline-block; }}
    </style><div class='grid'>{''.join(cards)}</div>"""
    st.iframe(doc, height=n_rows * height_per_row + 10)


def kpi_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def ms_caption(repo: Repository, query: str, version: str | None = None) -> None:
    ms = repo.last_query_ms.get(query)
    if ms is not None:
        st.caption(f"Consulta `{query}`: {ms:,.0f} ms" + (f" · modelo {version}" if version else ""))


def ts_to_date(ms: int) -> str:
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")
