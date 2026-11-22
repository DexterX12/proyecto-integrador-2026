"""Exporta el paquete de inferencia para Lambda (sin PyTorch) -> data/inference/model_version=<v>/

Contenido: faiss.index, faiss_item_ids.npy, faiss_meta.json (copiados del artefacto), user_mlp.npz
(pesos de la torre de usuario), asins.npy (parent_asin del catálogo, ordenado, dtype S10) y
manifest.json (hashes y dimensiones). La API trata al usuario como "sin ID" (vector de ID = 0), igual
que el modelo con usuarios sin embedding de ID.

Uso:  python -m jobs.export_inference_bundle --model-version <v>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil

import numpy as np

from src.common.config import add_common_args, load_config
from src.common.lake import Lake, ensure
from src.common.runlog import run_log
from src.data.dataset import ModelData
from src.models.train import load_model


def sha1(path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)
    md = ModelData(lake, cfg)
    out = ensure(lake.root / "inference" / f"model_version={args.model_version}")

    with run_log("export_inference_bundle", lake, {"model_version": args.model_version}) as run:
        model, config = load_model(lake, md, args.model_version)
        art = lake.artifacts(args.model_version)
        for name in ("faiss.index", "faiss_item_ids.npy", "faiss_meta.json"):
            shutil.copy2(art / name, out / name)

        sd = {k: v.numpy() for k, v in model.user_tower.state_dict().items()}
        n_linear = sorted({k.split(".")[1] for k in sd if k.startswith("mlp.") and k.endswith(".weight")}, key=int)
        assert len(n_linear) == 2, "El handler asume un MLP de 2 capas lineales (hidden_dims con un solo valor)"
        i0, i1 = n_linear
        np.savez(out / "user_mlp.npz", w0=sd[f"mlp.{i0}.weight"], b0=sd[f"mlp.{i0}.bias"],
                 w1=sd[f"mlp.{i1}.weight"], b1=sd[f"mlp.{i1}.bias"],
                 id_dim=np.int32(model.mcfg["id_emb_dim"]), cosine=np.bool_(model.cosine))

        asins = md.asin_of[1:]                               # item_idx = posición + 1
        assert (asins[:-1] <= asins[1:]).all(), "el catálogo debe estar ordenado por parent_asin"
        np.save(out / "asins.npy", np.array(asins.tolist(), dtype="S10"))

        manifest = {
            "model_version": args.model_version, "checkpoint_sha1": config["checkpoint_sha1"],
            "embedding_dim": config["embedding_dim"], "n_universe_items": config["n_universe_items"],
            "catalog_items": int(len(asins)), "history_len": md.H, "similarity": "cosine" if model.cosine else "dot",
            "files": {p.name: {"bytes": p.stat().st_size, "sha1": sha1(p)} for p in sorted(out.glob("*"))
                      if p.name != "manifest.json"},
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        total = sum(v["bytes"] for v in manifest["files"].values())
        run.metrics(bundle_dir=str(out), bundle_mb=round(total / 1e6, 1))


if __name__ == "__main__":
    main()
