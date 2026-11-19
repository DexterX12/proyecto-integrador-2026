"""Layout del data lake. Localmente es un directorio; en AWS es el mismo árbol bajo s3://bucket/.

    raw/ (archivos originales, no se copian ni modifican)
    curated/{reviews,products,interactions}/
    model_data/dataset_version=<v>/{train,validation,test,features,mappings}
    artifacts/model_version=<v>/
    recommendations/model_version=<v>/
    evaluation/model_version=<v>/
    published/current.json
    runs/  reports/
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import PROJECT_ROOT, Config


class Lake:
    def __init__(self, cfg: Config):
        root = os.environ.get("FASHREC_DATA_ROOT") or cfg.paths.data_root
        root_path = Path(root)
        self.root = root_path if root_path.is_absolute() else PROJECT_ROOT / root_path
        self.dataset_version = cfg.protocol.dataset_version
        self._cfg = cfg

    # --- RAW (se leen donde están; nunca se modifican) ---
    def raw(self, key: str) -> Path:
        p = Path(self._cfg.paths[key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    # --- CURATED ---
    def curated(self, table: str) -> Path:
        return self.root / "curated" / table

    # --- MODEL DATA ---
    @property
    def model_data(self) -> Path:
        return self.root / "model_data" / f"dataset_version={self.dataset_version}"

    # --- Versionados por modelo ---
    def artifacts(self, model_version: str) -> Path:
        return self.root / "artifacts" / f"model_version={model_version}"

    def recommendations(self, model_version: str) -> Path:
        return self.root / "recommendations" / f"model_version={model_version}"

    def evaluation(self, model_version: str) -> Path:
        return self.root / "evaluation" / f"model_version={model_version}"

    @property
    def published(self) -> Path:
        return self.root / "published"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def list_model_versions(self) -> list[str]:
        base = self.root / "artifacts"
        if not base.exists():
            return []
        return sorted(p.name.split("=", 1)[1] for p in base.glob("model_version=*"))


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
