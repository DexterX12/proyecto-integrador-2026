"""Taxonomía de productos derivada del título.

En Amazon Fashion 2023 el campo `categories` está vacío y `main_category` es constante
("AMAZON FASHION") para el 100 % de los productos, por lo que no sirven como feature
categórica. Se derivan `product_type` y `audience` con reglas de palabras clave.
La primera regla que coincide gana, así que el orden importa (p. ej. "dress shoes" -> shoes).

Las mismas reglas se usan en Spark (ETL) y en Python (inferencia de productos nuevos).
"""
from __future__ import annotations

import re

# (nombre, regex sobre el título en minúsculas). Sintaxis compatible con Java y Python.
PRODUCT_TYPES: list[tuple[str, str]] = [
    ("watch", r"\bwatch(es)?\b|\bsmartwatch"),
    ("eyewear", r"sunglasses?|\beyeglasses|\bglasses\b|\beyewear"),
    ("footwear", r"\bshoes?\b|\bboots?\b|sneakers?|sandals?|slippers?|loafers?|\bheels?\b|\bflats\b|\bpumps\b|\bclogs?\b|moccasins?|footwear|\bbooties\b|espadrilles?|flip flops?"),
    ("socks_hosiery", r"\bsocks?\b|stockings?|hosiery|\btights\b|pantyhose"),
    ("underwear", r"\bbras?\b|panty|panties|underwear|\bbriefs?\b|boxers?|lingerie|\bthongs?\b|bralette|shapewear|corset|undershirt|\bundies\b"),
    ("sleepwear", r"pajamas?|pyjamas?|nightgown|nightshirt|nightwear|\brobes?\b|sleepwear|loungewear|sleep set|onesie"),
    ("swimwear", r"swimsuit|bikini|swimwear|\bswim\b|swim trunks|trunks|rash guard|tankini|bathing suit|monokini|cover[- ]?up"),
    ("necklace", r"necklace|pendant|choker|\blocket\b|lariat"),
    ("earrings", r"earrings?|\bstuds\b|ear cuffs?|\bhoops?\b"),
    ("bracelet", r"bracelet|bangle|anklet|\bcuff\b|wristband"),
    ("ring", r"\brings?\b"),
    ("jewelry_other", r"jewelry|jewellery|brooch|\bcharms?\b|cufflinks?|\bpins?\b|body chain|piercing|\bbeads?\b"),
    ("bags", r"\bbags?\b|handbag|\bpurse|backpack|wallet|\btote\b|clutch|satchel|crossbody|luggage|fanny pack|messenger|\bpouch\b|briefcase"),
    ("headwear", r"\bhats?\b|\bcaps?\b|beanie|beret|fedora|visor|headwear|bucket hat"),
    ("scarf_gloves", r"scarf|scarves|\bgloves?\b|mittens?|shawl|\bwraps?\b|bandana|earmuffs?|neck gaiter|balaclava"),
    ("belt_tie", r"\bbelts?\b|suspenders|\bties?\b|bow tie|necktie|cummerbund"),
    ("costume", r"costume|cosplay|halloween|\bmask\b|\bwig\b|\btutu\b"),
    ("hair_accessory", r"headband|hair clip|hair tie|scrunchie|barrette|hair band|hairband|hair accessor|hair pin|\bhair\b"),
    ("dress", r"\bdress(es)?\b|\bgowns?\b|sundress|\bmaxi\b|\bkaftan|\bmuumuu"),
    ("jumpsuit", r"jumpsuit|romper|overalls?|coveralls?|playsuit|\bunitard"),
    ("skirt", r"\bskirts?\b|\bskort"),
    ("pants", r"\bpants\b|\bjeans\b|leggings?|trousers|joggers?|sweatpants|chinos?|jeggings|capris?\b|\bcargo\b|slacks|\bpant\b|culottes|palazzo|yoga pants|\bdenim\b"),
    ("shorts", r"\bshorts?\b|bermuda"),
    ("outerwear", r"jackets?|\bcoats?\b|\bparkas?\b|\bvests?\b|blazers?|windbreaker|trench|poncho|puffer|\bcapes?\b|bomber|anorak|overcoat|raincoat|\bsuits?\b"),
    ("sweater_hoodie", r"sweaters?|cardigans?|hoodies?|sweatshirts?|pullovers?|\bjumpers?\b|\bfleece\b|\bknit(ted|wear)?\b|\bhoody\b"),
    ("tshirt", r"t-shirts?|\btees?\b|\btshirts?\b|\btanks?\b|\bpolo\b|\bhenley"),
    ("shirt_top", r"\bshirts?\b|blouses?|\btops?\b|\btunics?\b|camisoles?|\bcami\b|bodysuit|\bcrop\b|\bbustier|\bhalter"),
    ("workwear", r"scrubs?|uniforms?|\bapron|lab coat|coveralls"),
    ("accessory_other", r"keychain|key chain|umbrella|sunglass|\bpatch\b|lanyard|\bsticker|\bgift\b|\bcase\b|phone"),
]
OTHER_TYPE = "other"

AUDIENCES: list[tuple[str, str]] = [
    ("baby", r"\bbaby\b|\binfants?\b|\bnewborn|\btoddlers?\b"),
    ("girls", r"\bgirls?\b|\bgirl's\b|\bgirls'"),
    ("boys", r"\bboys?\b|\bboy's\b|\bboys'"),
    ("kids", r"\bkids?\b|\bchildren|\bchild's\b|\byouth\b|\bjuniors?\b"),
    ("women", r"\bwomen'?s?\b|\bwoman\b|\bladies\b|\blady\b|\bfemale\b|\bmisses\b|\bwomens\b"),
    ("men", r"\bmen'?s?\b|\bman\b|\bgentlemen\b|\bmale\b|\bmens\b"),
]
UNKNOWN_AUDIENCE = "unisex_unknown"

PRODUCT_TYPE_VOCAB = [name for name, _ in PRODUCT_TYPES] + [OTHER_TYPE]
AUDIENCE_VOCAB = [name for name, _ in AUDIENCES] + [UNKNOWN_AUDIENCE]


def spark_case_expr(column, rules: list[tuple[str, str]], default: str):
    """Construye una expresión CASE WHEN encadenada para Spark."""
    from pyspark.sql import functions as F

    col = F.lower(F.coalesce(column, F.lit("")))
    expr = None
    for name, pattern in rules:
        cond = col.rlike(pattern)
        expr = F.when(cond, F.lit(name)) if expr is None else expr.when(cond, F.lit(name))
    return expr.otherwise(F.lit(default))


_COMPILED_TYPES = [(n, re.compile(p)) for n, p in PRODUCT_TYPES]
_COMPILED_AUD = [(n, re.compile(p)) for n, p in AUDIENCES]


def classify_title(title: str | None) -> tuple[str, str]:
    """Equivalente en Python para productos nuevos fuera del ETL."""
    text = (title or "").lower()
    ptype = next((n for n, rx in _COMPILED_TYPES if rx.search(text)), OTHER_TYPE)
    aud = next((n for n, rx in _COMPILED_AUD if rx.search(text)), UNKNOWN_AUDIENCE)
    return ptype, aud
