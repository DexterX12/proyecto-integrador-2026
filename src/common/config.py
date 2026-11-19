"""Carga de configuración YAML versionada con overrides por CLI (--set a.b=valor)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict con acceso por atributo (cfg.train.epochs) y hash estable."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - mensaje claro
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))

    def digest(self, *sections: str) -> str:
        """Hash corto de las secciones indicadas (o de toda la config)."""
        data = {s: self[s] for s in sections} if sections else dict(self)
        raw = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha1(raw).hexdigest()[:10]


def _coerce(value: str) -> Any:
    return yaml.safe_load(value)


def _set_path(tree: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"Clave de configuración inexistente: {dotted}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Clave de configuración inexistente: {dotted}")
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}'. Use clave.subclave=valor")
        key, raw = item.split("=", 1)
        _set_path(data, key.strip(), _coerce(raw.strip()))
    return Config(data)


def add_common_args(parser) -> None:
    parser.add_argument("--config", default=None, help="Ruta a un YAML de configuración")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="Overrides clave.subclave=valor")
