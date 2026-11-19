"""Job 6: publica una versión de resultados solo si pasa todas las validaciones.

Uso:
  python -m jobs.publish_job --model-version <v>
  python -m jobs.publish_job --rollback             # vuelve a la versión publicada anterior
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from src.common.config import add_common_args, load_config
from src.common.lake import Lake, ensure
from src.common.runlog import run_log
from src.data.dataset import ModelData
from src.data.ingestion import verify_raw_manifest
from src.evaluation.evaluate import check_index_consistency
from src.models.train import load_model
from src.retrieval.faiss_index import FlatIPIndex


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def validate_candidate(cfg, lake, md, version: str, require_beat_baseline: bool = True) -> list[dict]:
    """Devuelve la lista de chequeos (nombre, ok, detalle)."""
    checks: list[dict] = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    art, rec, ev = lake.artifacts(version), lake.recommendations(version), lake.evaluation(version)
    add("artefactos existen", all((art / f).exists() for f in
                                  ("model.pt", "config.json", "faiss.index", "training_history.csv")), art)
    manifest = lake.reports / "raw_manifest.json"
    if manifest.exists():
        ok = verify_raw_manifest({"reviews": lake.raw("raw_reviews"), "metadata": lake.raw("raw_metadata")}, manifest)
        add("datos RAW sin modificar (SHA-256)", ok)
    else:
        add("datos RAW sin modificar (SHA-256)", False, "falta raw_manifest.json")
    try:
        model, config = load_model(lake, md, version)
        add("checkpoint coincide con config.json y protocolo", True, config["checkpoint_sha1"][:12])
        index = FlatIPIndex.load(art)
        cons = check_index_consistency(model, index, md)
        add("índice FAISS corresponde al checkpoint y mappings", cons["ok"], cons)
        mapping_ok = all(
            __import__("src.models.train", fromlist=["sha1_file"]).sha1_file(art / "id_mappings" / n) == h
            for n, h in config["mappings_sha1"].items())
        add("mappings versionados intactos", mapping_ok)
    except Exception as exc:  # noqa: BLE001
        add("checkpoint/índice cargan correctamente", False, exc)
        config = None

    val = _read(rec / "validation.json")
    add("recomendaciones generadas", val is not None)
    if val:
        add("Top-N sin productos vistos en el contexto", val["recommended_seen_items_violations"] == 0,
            f"violaciones={val['recommended_seen_items_violations']}")
        add("listas completas, sin duplicados, ordenadas por score",
            val["users_with_less_than_top_n"] == 0 and val["duplicates_within_list"] == 0 and val["scores_sorted_desc"])
        add("todos los ítems recomendados están en el universo", val["all_items_in_universe"])
        add("recomendaciones etiquetadas con la versión del modelo", val["model_version"] == version)

    metrics = _read(ev / "metrics.json")
    add("evaluación en validation y test disponible",
        metrics is not None and {"validation", "test"} <= set(metrics.get("splits", {})))
    if metrics and {"validation", "test"} <= set(metrics["splits"]):
        add("evaluación usa el mismo protocolo del dataset",
            metrics["meta"]["protocol_digest"] == md.manifest["protocol_digest"])
        add("Two-Tower evaluado", "two_tower" in metrics["splits"]["test"])
        if require_beat_baseline and "two_tower" in metrics["splits"]["validation"]:
            tt = metrics["splits"]["validation"]["two_tower"]["overall"]["ndcg@20"]
            pop = metrics["splits"]["validation"]["popularity"]["overall"]["ndcg@20"]
            add("supera a la línea base de popularidad (NDCG@20 validation)", tt > pop,
                f"two_tower={tt:.4f} popularity={pop:.4f}")
        current = _read(lake.published / "current.json")
        if current and current["model_version"] != version:
            prev = _read(lake.evaluation(current["model_version"]) / "metrics.json")
            if prev:
                prev_v = prev["splits"]["validation"]["two_tower"]["overall"]["ndcg@20"]
                new_v = metrics["splits"]["validation"]["two_tower"]["overall"]["ndcg@20"]
                add("no empeora la versión publicada (NDCG@20 validation)", new_v >= prev_v * 0.98,
                    f"nueva={new_v:.4f} actual={prev_v:.4f}")
    return checks


def publish(cfg, lake, md, version: str, require_beat_baseline: bool, force: bool) -> dict:
    checks = validate_candidate(cfg, lake, md, version, require_beat_baseline)
    failed = [c for c in checks if not c["ok"]]
    report = {"model_version": version, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "checks": checks, "passed": not failed}
    ensure(lake.published)
    (lake.published / f"validation_{version}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for c in checks:
        print(("  [OK]   " if c["ok"] else "  [FALLA]"), c["check"], "-", c["detail"][:110] if c["detail"] else "")
    if failed and not force:
        print(f"\nNO se publica {version}: {len(failed)} chequeo(s) fallaron. Se conserva la versión actual.")
        return report
    metrics = _read(lake.evaluation(version) / "metrics.json") or {}
    current = _read(lake.published / "current.json")
    history = _read(lake.published / "history.json") or []
    entry = {
        "model_version": version, "dataset_version": md.manifest["dataset_version"],
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "previous_model_version": current["model_version"] if current else None,
        "forced": bool(failed),
    }
    (lake.published / "current.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    history.append(entry)
    (lake.published / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nPublicado: {version} (anterior: {entry['previous_model_version']})")
    return report


def rollback(lake) -> None:
    history = _read(lake.published / "history.json") or []
    current = _read(lake.published / "current.json")
    if not current or not current.get("previous_model_version"):
        sys.exit("No hay versión anterior a la cual volver.")
    prev = current["previous_model_version"]
    entry = {**current, "model_version": prev, "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "previous_model_version": current["model_version"], "rollback": True}
    (lake.published / "current.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    history.append(entry)
    (lake.published / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Rollback: {current['model_version']} -> {prev}")


def main(argv=None) -> dict | None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-version")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--force", action="store_true", help="Publica aunque fallen chequeos (se registra)")
    parser.add_argument("--no-baseline-gate", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)
    if args.rollback:
        rollback(lake)
        return None
    if not args.model_version:
        parser.error("--model-version es obligatorio (o use --rollback)")
    md = ModelData(lake, cfg)
    with run_log("publish", lake, {"model_version": args.model_version}) as run:
        report = publish(cfg, lake, md, args.model_version, not args.no_baseline_gate, args.force)
        run.metric("passed", report["passed"])
    return report


if __name__ == "__main__":
    main()
