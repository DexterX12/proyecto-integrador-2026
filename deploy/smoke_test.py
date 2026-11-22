"""Pruebas funcionales de la API desplegada (criterios 6 a 8 de DEPLOYMENT_SERVERLESS.md §5).

Compara las respuestas con los resultados locales (data/recommendations) y mide latencias; las capturas de
esta salida y de la consola de API Gateway/Lambda/CloudWatch sirven como evidencia.

Uso:  python deploy/smoke_test.py --api-url https://xxxx.execute-api.us-east-1.amazonaws.com [--model-version V]
Código de salida 0 solo si todas las pruebas pasan.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool, str]] = []


def call(method: str, url: str, body: dict | str | None = None) -> tuple[int, dict, float]:
    data = None
    headers = {}
    if body is not None:
        data = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            code, raw = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        code, raw = err.code, err.read()
    ms = (time.perf_counter() - t0) * 1000
    try:
        return code, json.loads(raw), ms
    except ValueError:
        return code, {"raw": raw[:200].decode("utf-8", "replace")}, ms


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'OK' if ok else 'FALLA'}] {name}: {detail}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--model-version")
    args = ap.parse_args(argv)
    api = args.api_url.rstrip("/")
    data = Path(args.data_root)
    version = args.model_version or json.loads((data / "published" / "current.json").read_text(encoding="utf-8"))["model_version"]

    recs = pq.read_table(data / "recommendations" / f"model_version={version}" / "recommendations.parquet",
                         columns=["user_id", "rank", "parent_asin"]).slice(0, 500).to_pandas()
    user = recs["user_id"].iloc[0]
    expected = recs[recs["user_id"] == user].sort_values("rank")["parent_asin"].tolist()
    history = expected[:3]
    print(f"API: {api}\nVersión local de referencia: {version} | usuario de prueba: {user}\n")

    code, body, ms = call("GET", f"{api}/v1/health")
    check("GET /v1/health", code == 200 and body.get("model_version") == version,
          f"{code} model_version={body.get('model_version')} ({ms:.0f} ms)")

    code, body, ms = call("GET", f"{api}/v1/recommendations/{user}?n=10")
    got = [i["parent_asin"] for i in body.get("items", [])]
    check("GET /v1/recommendations/{user}: igual a recommendations.parquet",
          code == 200 and body.get("source") == "precomputed" and got == expected[:10],
          f"{code} source={body.get('source')} top10_igual={got == expected[:10]} ({ms:.0f} ms)")
    first = (body.get("items") or [{}])[0]
    check("respuesta enriquecida con título e imagen", bool(first.get("title")) and bool(first.get("image_url")),
          f"{str(first.get('title'))[:50]!r}")

    code, body, ms = call("GET", f"{api}/v1/recommendations/USUARIO_QUE_NO_EXISTE?n=5")
    check("usuario desconocido -> fallback por popularidad",
          code == 200 and body.get("source") == "popularity_fallback" and len(body.get("items", [])) == 5,
          f"{code} source={body.get('source')} items={len(body.get('items', []))} ({ms:.0f} ms)")

    code, body, ms = call("GET", f"{api}/v1/recommendations/{user}?n=abc")
    check("n inválido -> 400", code == 400, f"{code} {body.get('error')}")

    code, body, cold = call("POST", f"{api}/v1/recommend", {"history": history, "n": 10})
    asins = [i.get("parent_asin") for i in body.get("items", [])]
    check("POST /v1/recommend (arranque en frío)", code == 200 and body.get("source") == "model" and len(asins) == 10,
          f"{code} source={body.get('source')} items={len(asins)} ({cold:.0f} ms)")
    check("el Top-N no repite productos del historial", not set(asins) & set(history), f"historial={history}")
    code, body, warm = call("POST", f"{api}/v1/recommend", {"history": history, "n": 10})
    check("POST /v1/recommend (instancia caliente)", code == 200 and [i.get("parent_asin") for i in body.get("items", [])] == asins,
          f"{code} misma respuesta que la primera ({warm:.0f} ms)")

    code, body, _ = call("POST", f"{api}/v1/recommend", {"history": ["NOEXISTE01", history[0]], "n": 5})
    check("producto desconocido se ignora y se reporta", code == 200 and body.get("unknown_asins") == ["NOEXISTE01"],
          f"{code} unknown_asins={body.get('unknown_asins')}")
    code, body, _ = call("POST", f"{api}/v1/recommend", {"history": ["NOEXISTE01"], "n": 5})
    check("sin historial utilizable -> lista vacía", code == 200 and body.get("source") == "no_usable_history" and not body.get("items"),
          f"{code} source={body.get('source')}")
    code, body, _ = call("POST", f"{api}/v1/recommend", "esto no es json")
    check("body inválido -> 400", code == 400, f"{code}")
    code, body, _ = call("POST", f"{api}/v1/recommend", {"history": []})
    check("history vacío -> 400", code == 400, f"{code}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} pruebas OK" + ("" if not failed else f"; fallaron: {[r[0] for r in failed]}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
