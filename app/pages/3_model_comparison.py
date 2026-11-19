import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.common import get_repo, sidebar_version  # noqa: E402

st.set_page_config(page_title="Modelos · FashRec", page_icon="⚖️", layout="wide")
repo = get_repo()
version = sidebar_version(repo)
st.title("⚖️ Comparación de modelos")
if version is None:
    st.stop()

metrics = repo.metrics(version)
if not metrics:
    st.error("No hay métricas para esta versión.")
    st.stop()
meta = metrics["meta"]
cut = meta["cutoffs"]
st.caption(f"Modelo `{version}` · dataset `{meta['dataset_version']}` · relevante = rating ≥ {meta['relevance_threshold']:.0f} · "
           f"train < {cut['val_start']} ≤ validation < {cut['test_start']} ≤ test · "
           f"universo candidato: {meta['candidate_universe_items']:,} productos · mismos usuarios, universo y filtro de vistos para todos los modelos.")

with st.sidebar:
    st.header("Filtros")
    split = st.radio("Partición", [s for s in ("test", "validation") if s in metrics["splits"]],
                     help="Test se usa como evaluación final; validation se usó para ajustar.")
    ks = meta["ks"]
    k = st.selectbox("K", ks, index=len(ks) - 1)
    segments = ["todos"] + sorted(next(iter(metrics["splits"][split].values()))["segments"].keys())
    segment = st.selectbox("Segmento de usuarios", segments,
                           help="hist_N = nº de productos positivos previos; user_id_known = usuarios con embedding de ID.")

NAMES = {"popularity": "Popularidad", "item_knn": "Item-KNN", "two_tower": "Two-Tower (FashRec)"}
rows = []
for model, res in metrics["splits"][split].items():
    src = res["overall"] if segment == "todos" else res["segments"].get(segment, {})
    n_users = res["n_users"] if segment == "todos" else src.get("n_users", 0)
    ba = res["beyond_accuracy"].get(f"@{k}", {})
    rows.append({"Modelo": NAMES.get(model, model), "Usuarios": n_users,
                 f"Recall@{k}": src.get(f"recall@{k}"), f"NDCG@{k}": src.get(f"ndcg@{k}"),
                 f"Precision@{k}": res["overall"].get(f"precision@{k}") if segment == "todos" else None,
                 f"MAP@{k}": res["overall"].get(f"map@{k}") if segment == "todos" else None,
                 "Cobertura": ba.get("coverage"), "Diversidad": ba.get("diversity"), "Novedad": ba.get("novelty")})
table = pd.DataFrame(rows)

c1, c2 = st.columns(2)
for col, metric in ((c1, f"Recall@{k}"), (c2, f"NDCG@{k}")):
    with col:
        st.subheader(metric)
        fig = px.bar(table, x="Modelo", y=metric, color="Modelo", text_auto=".4f")
        if segment == "todos":
            cis = [metrics["splits"][split][m].get("ci95", {}).get(f"{metric.split('@')[0].lower()}@20") for m in
                   metrics["splits"][split]]
            if k == 20 and all(cis):
                fig.update_traces(error_y=dict(type="data", symmetric=False,
                                               array=[c[1] - v for c, v in zip(cis, table[metric])],
                                               arrayminus=[v - c[0] for c, v in zip(cis, table[metric])]))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
if k == 20 and segment == "todos":
    st.caption("Barras de error: IC 95 % por bootstrap sobre usuarios.")

st.subheader("Tabla comparativa")
st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns if c not in ("Modelo", "Usuarios")}, na_rep="—"),
             width="stretch", hide_index=True)

# curva por K
st.subheader("Recall y NDCG según K")
curve = []
for model, res in metrics["splits"][split].items():
    for kk in ks:
        src = res["overall"] if segment == "todos" else res["segments"].get(segment, {})
        for m in ("recall", "ndcg"):
            if f"{m}@{kk}" in src:
                curve.append({"Modelo": NAMES.get(model, model), "K": kk, "Métrica": m.upper(), "valor": src[f"{m}@{kk}"]})
if curve:
    st.plotly_chart(px.line(pd.DataFrame(curve), x="K", y="valor", color="Modelo", facet_col="Métrica", markers=True),
                    width="stretch")

st.subheader("Segmentos de usuarios (Recall@20 / NDCG@20)")
seg_rows = []
for model, res in metrics["splits"][split].items():
    for seg, vals in res["segments"].items():
        seg_rows.append({"Modelo": NAMES.get(model, model), "Segmento": seg, "Usuarios": vals["n_users"],
                         "Recall@20": vals.get("recall@20"), "NDCG@20": vals.get("ndcg@20")})
seg_df = pd.DataFrame(seg_rows)
if not seg_df.empty:
    st.plotly_chart(px.bar(seg_df, x="Segmento", y="Recall@20", color="Modelo", barmode="group", hover_data=["Usuarios"]),
                    width="stretch")

# cold start
if metrics.get("cold_start"):
    st.subheader("Cold start de ítems (test, candidatos = train + validation)")
    cm = metrics["cold_start_meta"]
    st.caption(f"{cm['cold_items_added']:,} productos candidatos nuevos (sin interacciones en train) y "
               f"{cm['cold_targets']:,} compras relevantes sobre ellos. Two-Tower los puntúa con atributos de contenido (ID desconocido).")
    cold_rows = []
    for model, res in metrics["cold_start"].items():
        cold_rows.append({"Modelo": NAMES.get(model, model), "Recall@20 (todos los relevantes)": res["overall"]["recall@20"],
                          "Recall@20 solo ítems nuevos": res["cold_items"]["cold_recall@20"],
                          "Aciertos nuevos": res["cold_items"]["cold_hits@20"],
                          "% del Top-20 que son ítems nuevos": res["cold_items"]["share_of_top20_slots_cold_items"]})
    st.dataframe(pd.DataFrame(cold_rows).style.format({c: "{:.4f}" for c in list(pd.DataFrame(cold_rows).columns)[1:3]}
                                                      | {"% del Top-20 que son ítems nuevos": "{:.2%}"}),
                 width="stretch", hide_index=True)

hist = repo.training_history(version)
if not hist.empty:
    st.subheader("Historia de entrenamiento")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="loss (train)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_ndcg@20"], name="NDCG@20 (validation)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_recall@20"], name="Recall@20 (validation)", yaxis="y2"))
    fig.update_layout(xaxis_title="época", yaxis=dict(title="loss"), yaxis2=dict(title="métrica", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")
    st.caption("El checkpoint se elige por NDCG@20 en validation (subconjunto de usuarios); test no se usa para decidir.")

with st.expander("Detalles del protocolo"):
    st.json({"eval_sets": meta["eval_sets"], "baselines": meta["baselines"], "faiss_consistency": meta["faiss_consistency"]})
