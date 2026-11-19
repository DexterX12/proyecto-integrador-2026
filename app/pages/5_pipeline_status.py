import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.common import get_repo, kpi_row, sidebar_version  # noqa: E402

st.set_page_config(page_title="Pipeline · FashRec", page_icon="🛠️", layout="wide")
repo = get_repo()
version = sidebar_version(repo)
st.title("🛠️ Estado del pipeline")

pub = repo.published()
runs = repo.runs()
if pub:
    kpi_row([("Versión publicada", pub["model_version"]), ("Publicada (UTC)", pub["published_at"][:16].replace("T", " ")),
             ("Versión anterior (rollback)", pub.get("previous_model_version") or "—"),
             ("Ejecuciones registradas", str(len(runs)))])
else:
    st.warning("Todavía no se publicó ninguna versión.")

st.subheader("Última ejecución de cada job")
if runs.empty:
    st.info("Sin ejecuciones registradas.")
else:
    last = runs.sort_values("started_at").groupby("job").tail(1)
    last = last[~last["job"].str.startswith("tune_")]
    last["estado"] = last["status"].map({"success": "✅ success", "failed": "❌ failed", "running": "⏳ running"})
    st.dataframe(last[["job", "estado", "started_at", "duration_s", "error"]].rename(
        columns={"started_at": "inicio (UTC)", "duration_s": "duración (s)"}), width="stretch", hide_index=True)
    fig = px.bar(last, x="job", y="duration_s", color="status", labels={"duration_s": "segundos"})
    st.plotly_chart(fig, width="stretch")
    with st.expander("Conteos de entrada/salida registrados por job"):
        for r in last.itertuples():
            st.markdown(f"**{r.job}**")
            st.json({k: v for k, v in r.metrics.items() if not isinstance(v, dict) or len(str(v)) < 800})

st.subheader("Chequeos de publicación")
if version:
    f = repo.lake.published / f"validation_{version}.json"
    if f.exists():
        rep = json.loads(f.read_text(encoding="utf-8"))
        chk = pd.DataFrame(rep["checks"])
        chk["ok"] = chk["ok"].map({True: "✅", False: "❌"})
        st.caption(f"Versión `{version}` · {'aprobada' if rep['passed'] else 'NO aprobada'} · {rep['checked_at'][:16].replace('T', ' ')} UTC")
        st.dataframe(chk, width="stretch", hide_index=True)
    else:
        st.info("Esta versión aún no pasó por publish_job.")

st.subheader("Calidad del ETL")
etl = repo.etl_report()
if etl:
    c1, c2 = st.columns(2)
    c1.json({"raw": etl["raw"], "reviews": etl["reviews"]})
    c2.json({"interactions": etl["interactions"], "products_null_rate": etl["products"]["null_rate"]})

st.subheader("Historial de publicaciones")
hist = repo.publish_history()
if hist:
    st.dataframe(pd.DataFrame(hist)[::-1], width="stretch", hide_index=True)

st.subheader("Rendimiento de consultas del dashboard")
log = repo.serving_log(300)
if log.empty:
    st.info("Aún no hay consultas registradas.")
else:
    summary = log.groupby("query").agg(n=("ms", "size"), p50_ms=("ms", "median"), p95_ms=("ms", lambda s: s.quantile(.95)),
                                       errores=("status", lambda s: int((s == "error").sum()))).reset_index()
    st.dataframe(summary.round(1), width="stretch", hide_index=True)
    errs = log[log["status"] == "error"]
    if not errs.empty:
        st.error("Errores recientes de lectura")
        st.dataframe(errs[["ts", "query", "error"]], width="stretch", hide_index=True)
