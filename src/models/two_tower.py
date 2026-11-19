"""Modelo Two-Tower híbrido (PyTorch).

Torre de producto : embedding de ID + categoría + audiencia + tienda + precio + texto (TF-IDF->SVD) -> MLP.
Torre de usuario  : concat(embedding de user_id, media de los embeddings de los productos del historial) -> MLP.
Score             : coseno (producto punto de vectores L2-normalizados) / temperatura.

El ID desconocido (índice 0) se entrena con "ID dropout", de modo que la torre de producto produce un
vector útil solo con contenido (cold start de ítems) y la de usuario con solo historial (usuario sin ID).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ItemFeatures:
    """Tensores de features por item_idx (fila 0 = padding). No forman parte del state_dict."""
    id_idx: torch.Tensor
    product_type: torch.Tensor
    audience: torch.Tensor
    store: torch.Tensor
    price: torch.Tensor
    has_price: torch.Tensor
    text: torch.Tensor
    log_pop: torch.Tensor

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "ItemFeatures":
        return cls(
            id_idx=torch.from_numpy(arrays["id_idx"].astype(np.int64)),
            product_type=torch.from_numpy(arrays["product_type"].astype(np.int64)),
            audience=torch.from_numpy(arrays["audience"].astype(np.int64)),
            store=torch.from_numpy(arrays["store"].astype(np.int64)),
            price=torch.from_numpy(arrays["price"].astype(np.float32)),
            has_price=torch.from_numpy(arrays["has_price"].astype(np.float32)),
            text=torch.from_numpy(arrays["text"].astype(np.float32)),
            log_pop=torch.from_numpy(arrays["log_pop"].astype(np.float32)),
        )

    def vocab_sizes(self) -> dict[str, int]:
        return {
            "n_id": int(self.id_idx.max()) + 1,
            "n_type": int(self.product_type.max()) + 1,
            "n_aud": int(self.audience.max()) + 1,
            "n_store": int(self.store.max()) + 1,
            "text_dim": int(self.text.shape[1]),
        }


def _init_embedding(emb: nn.Embedding, std: float) -> None:
    """N(0, std) con std pequeño: con el N(0,1) por defecto los IDs (ruido al inicio) tapan el contenido."""
    nn.init.normal_(emb.weight, mean=0.0, std=std)
    with torch.no_grad():
        emb.weight[0].zero_()


def _mlp(in_dim: int, hidden: list[int], out_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ItemTower(nn.Module):
    def __init__(self, feats: ItemFeatures, mcfg: dict, sizes: dict):
        super().__init__()
        self.feats = feats
        self.id_dropout = float(mcfg["item_id_dropout"])
        self.use_pop = bool(mcfg.get("use_popularity_feature", True))
        self.id_emb = nn.Embedding(sizes["n_id"], mcfg["id_emb_dim"], padding_idx=0)
        self.type_emb = nn.Embedding(sizes["n_type"], mcfg["cat_emb_dim"], padding_idx=0)
        self.aud_emb = nn.Embedding(sizes["n_aud"], mcfg["cat_emb_dim"], padding_idx=0)
        self.store_emb = nn.Embedding(sizes["n_store"], mcfg["store_emb_dim"], padding_idx=0)
        in_dim = (mcfg["id_emb_dim"] + 2 * mcfg["cat_emb_dim"] + mcfg["store_emb_dim"]
                  + 2 + sizes["text_dim"] + int(self.use_pop))
        self.mlp = _mlp(in_dim, list(mcfg["hidden_dims"]), mcfg["embedding_dim"], mcfg["dropout"])
        for emb in (self.id_emb, self.type_emb, self.aud_emb, self.store_emb):
            _init_embedding(emb, mcfg.get("emb_init_std", 0.1))

    def forward(self, item_idx: torch.Tensor, use_id: bool = True) -> torch.Tensor:
        f = self.feats
        ids = f.id_idx[item_idx]
        if self.training and self.id_dropout > 0:
            ids = ids.masked_fill(torch.rand(ids.shape) < self.id_dropout, 0)
        if not use_id:
            ids = torch.zeros_like(ids)
        parts = [
            self.id_emb(ids), self.type_emb(f.product_type[item_idx]), self.aud_emb(f.audience[item_idx]),
            self.store_emb(f.store[item_idx]), f.price[item_idx, None], f.has_price[item_idx, None],
            f.text[item_idx],
        ]
        if self.use_pop:
            parts.append(f.log_pop[item_idx, None])
        x = torch.cat(parts, dim=1)
        return self.mlp(x)


class UserTower(nn.Module):
    def __init__(self, n_users: int, mcfg: dict):
        super().__init__()
        self.user_id_dropout = float(mcfg["user_id_dropout"])
        self.uid_emb = nn.Embedding(n_users, mcfg["id_emb_dim"], padding_idx=0)
        self.mlp = _mlp(mcfg["id_emb_dim"] + mcfg["embedding_dim"], list(mcfg["hidden_dims"]),
                        mcfg["embedding_dim"], mcfg["dropout"])
        _init_embedding(self.uid_emb, mcfg.get("emb_init_std", 0.1))

    def forward(self, user_idx: torch.Tensor, hist_vec: torch.Tensor) -> torch.Tensor:
        if self.training and self.user_id_dropout > 0:
            user_idx = user_idx.masked_fill(torch.rand(user_idx.shape) < self.user_id_dropout, 0)
        return self.mlp(torch.cat([self.uid_emb(user_idx), hist_vec], dim=1))


class TwoTower(nn.Module):
    def __init__(self, feats: ItemFeatures, n_users: int, mcfg: dict):
        super().__init__()
        self.mcfg = dict(mcfg)
        self.cosine = self.mcfg.get("similarity", "cosine") == "cosine"
        self.sizes = feats.vocab_sizes()
        self.n_users = n_users
        self.item_tower = ItemTower(feats, self.mcfg, self.sizes)
        self.user_tower = UserTower(n_users, self.mcfg)

    # ---- inferencia -----------------------------------------------------------------------------
    @torch.no_grad()
    def encode_items(self, item_idx: np.ndarray | torch.Tensor, batch: int = 65536,
                     use_id: bool = True) -> np.ndarray:
        """Embeddings normalizados L2 de productos (ítems nuevos: el ID ya es OOV en id_idx)."""
        self.eval()
        idx = torch.as_tensor(item_idx, dtype=torch.long)
        out = [self._maybe_norm(self.item_tower(idx[i:i + batch], use_id=use_id))
               for i in range(0, len(idx), batch)]
        return torch.cat(out).numpy()

    def _maybe_norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.cosine else x

    def _history_vector(self, hist: torch.Tensor, item_emb_fn) -> torch.Tensor:
        mask = (hist > 0)
        uniq, inv = torch.unique(hist.flatten(), return_inverse=True)
        emb = F.normalize(item_emb_fn(uniq), dim=1)[inv].view(*hist.shape, -1)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        return (emb * mask.unsqueeze(-1)).sum(1) / denom

    @torch.no_grad()
    def encode_users(self, user_idx: np.ndarray, hist: np.ndarray, batch: int = 8192) -> np.ndarray:
        self.eval()
        u = torch.as_tensor(user_idx, dtype=torch.long)
        h = torch.as_tensor(hist, dtype=torch.long)
        out = []
        for i in range(0, len(u), batch):
            hv = self._history_vector(h[i:i + batch], self.item_tower)
            out.append(self._maybe_norm(self.user_tower(u[i:i + batch], hv)))
        return torch.cat(out).numpy()

    # ---- entrenamiento --------------------------------------------------------------------------
    def forward_train(self, user_idx: torch.Tensor, pos_item: torch.Tensor, hist: torch.Tensor):
        """Una pasada para positivos e historial (los ítems únicos se codifican una sola vez)."""
        b = pos_item.shape[0]
        all_items = torch.cat([pos_item, hist.flatten()])
        uniq, inv = torch.unique(all_items, return_inverse=True)
        emb = self.item_tower(uniq)[inv]
        pos_emb = emb[:b]
        h_emb = F.normalize(emb[b:], dim=1).view(b, hist.shape[1], -1)
        mask = (hist > 0).unsqueeze(-1)
        hv = (h_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        z_u = self._maybe_norm(self.user_tower(user_idx, hv))
        z_i = self._maybe_norm(pos_emb)
        return z_u, z_i


def contrastive_loss(z_u: torch.Tensor, z_i: torch.Tensor, pos_item: torch.Tensor, hist: torch.Tensor,
                     user_idx: torch.Tensor, temperature: float, log_q: torch.Tensor | None = None,
                     mask_history: bool = True) -> tuple[torch.Tensor, dict]:
    """Softmax contrastivo con negativos in-batch.

    Enmascara (excluye de los negativos) falsos negativos evidentes: el mismo ítem positivo repetido en
    el batch, ítems del historial del propio usuario y otros ejemplos del mismo usuario.
    `log_q` (log de la prob. de muestreo de cada ítem) corrige el sesgo de popularidad (logQ).
    """
    logits = (z_u @ z_i.T) / temperature
    if log_q is not None:
        logits = logits - log_q[pos_item][None, :]
    b = logits.shape[0]
    diag = torch.eye(b, dtype=torch.bool)
    false_neg = (pos_item[None, :] == pos_item[:, None])
    if mask_history:
        false_neg |= (hist[:, :, None] == pos_item[None, None, :]).any(1) & (hist.sum(1, keepdim=True) > 0)
    same_user = (user_idx[:, None] == user_idx[None, :]) & (user_idx[:, None] > 0)
    false_neg = (false_neg | same_user) & ~diag
    logits = logits.masked_fill(false_neg, float("-inf"))
    labels = torch.arange(b)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
        masked = false_neg.float().mean().item()
    return loss, {"in_batch_acc": acc, "masked_frac": masked}


