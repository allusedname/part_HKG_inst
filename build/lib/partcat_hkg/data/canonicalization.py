from __future__ import annotations

import re

OBJECT_PREFIX_WORDS = {
    "aeroplane", "airplane", "aircraft", "plane", "bird", "biped", "quadruped",
    "car", "bus", "train", "motorbike", "motorcycle", "bicycle", "bike", "boat",
    "bottle", "person", "human", "animal", "dog", "cat", "horse", "cow", "sheep",
    "vehicle", "fish", "reptile", "snake",
}

QUALIFIER_WORDS = {
    "left", "right", "front", "back", "rear", "upper", "lower", "inner", "outer",
    "top", "bottom", "side",
}

PHRASE_TO_FUNCTION = {
    "side mirror": "mirror", "side view mirror": "mirror", "wing mirror": "mirror",
    "rear view mirror": "mirror", "license plate": "license_plate", "number plate": "license_plate",
    "body frame": "body", "body/frame": "body", "front wheel": "wheel", "rear wheel": "wheel",
    "tail fin": "tail_fin", "dorsal fin": "fin", "pectoral fin": "fin", "wind shield": "window",
    "windshield": "window", "head light": "headlight", "headlight": "headlight",
    "tail light": "taillight", "taillight": "taillight",
}

TOKEN_TO_FUNCTION = {
    "wings": "wing", "wing": "wing", "fin": "fin", "fins": "fin",
    "wheel": "wheel", "wheels": "wheel", "tire": "wheel", "tires": "wheel",
    "tyre": "wheel", "tyres": "wheel", "tier": "wheel", "tiers": "wheel",
    "leg": "leg", "legs": "leg", "foot": "foot", "feet": "foot", "paw": "foot",
    "paws": "foot", "hand": "hand", "hands": "hand", "arm": "arm", "arms": "arm",
    "ear": "ear", "ears": "ear", "eye": "eye", "eyes": "eye", "head": "head",
    "heads": "head", "tail": "tail", "tails": "tail", "door": "door", "doors": "door",
    "window": "window", "windows": "window", "mirror": "mirror", "mirrors": "mirror",
    "beak": "beak", "beaks": "beak", "body": "body", "torso": "body", "frame": "body",
    "fuselage": "body", "hull": "body", "trunk": "body", "neck": "neck", "nose": "nose",
    "mouth": "mouth", "horn": "horn", "horns": "horn", "seat": "seat", "saddle": "seat",
    "handlebar": "handlebar", "handlebars": "handlebar", "engine": "engine",
    "propeller": "propeller", "mast": "mast", "sail": "sail", "roof": "roof",
}

DIRECT_SYNONYM_TO_CANON = {"body/frame": "body", "side-view mirror": "mirror", "tier": "wheel"}
OBJECT_SYNONYM_TO_CANON = {
    "airplane": "aeroplane", "aircraft": "aeroplane", "plane": "aeroplane",
    "bike": "bicycle", "motorcycle": "motorbike", "person": "biped", "human": "biped",
}


def canonicalize_object_name(name: str) -> str:
    s = str(name).strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    s = " ".join(s.split())
    return OBJECT_SYNONYM_TO_CANON.get(s, s.replace(" ", "_"))


def display_object_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def _normalize_part_text(name: str) -> str:
    s = str(name).strip().lower().replace("_", " ").replace("-", " ").replace("/", " /")
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return " ".join(s.split())


def _singularize_token(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 3:
        return tok[:-3] + "y"
    if tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def canonicalize_part_name(name: str) -> str:
    s = _normalize_part_text(name)
    if s in DIRECT_SYNONYM_TO_CANON:
        return DIRECT_SYNONYM_TO_CANON[s]
    for phrase, canon in sorted(PHRASE_TO_FUNCTION.items(), key=lambda kv: -len(kv[0])):
        if phrase in s:
            return canon
    tokens = [_singularize_token(t) for t in s.replace("/", " ").split()]
    stripped = [t for t in tokens if t not in OBJECT_PREFIX_WORDS and t not in QUALIFIER_WORDS]
    stripped_s = " ".join(stripped)
    for phrase, canon in sorted(PHRASE_TO_FUNCTION.items(), key=lambda kv: -len(kv[0])):
        if phrase in stripped_s:
            return canon
    for tok in reversed(stripped):
        if tok in TOKEN_TO_FUNCTION:
            return TOKEN_TO_FUNCTION[tok]
    for tok in reversed(tokens):
        if tok in TOKEN_TO_FUNCTION:
            return TOKEN_TO_FUNCTION[tok]
    return stripped_s if stripped_s else s


def role_name(obj_name: str, func_name: str) -> str:
    return f"{canonicalize_object_name(obj_name)}:{str(func_name).strip().lower()}"


def prompt_part_text(func_name: str) -> str:
    aliases = {
        "mirror": "side mirror", "body": "body", "tail_fin": "tail fin",
        "license_plate": "license plate", "taillight": "tail light",
    }
    return aliases.get(str(func_name), str(func_name).replace("_", " "))


def role_prompt_text(role: str) -> tuple[str, str]:
    if ":" not in role:
        return "object", prompt_part_text(role)
    obj, part = role.split(":", 1)
    return obj.replace("_", " "), prompt_part_text(part)
