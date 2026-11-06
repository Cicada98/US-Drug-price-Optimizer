"""Split NADAC ``NDC Description`` strings into analysable columns.

NADAC descriptions follow a loose pattern::

    <DRUG NAME> [<RELEASE TYPE>] <STRENGTH> [<DOSAGE FORM>]

    ABILIFY 10 MG TABLET              -> ABILIFY | 10 MG | TABLET
    FOCALIN XR 20 MG CAPSULE          -> FOCALIN | XR | 20 MG | CAPSULE
    PERCOCET 5-325 MG TABLET          -> combination product, 2 ingredients
    ERYPED 200 MG/5 ML SUSPENSION     -> concentration: 200 MG per 5 ML
    BUTRANS 10 MCG/HR PATCH           -> delivery rate: 10 MCG per 1 HR
    LOTEMAX 0.5% EYE DROPS            -> percentage strength
    FLEET ENEMA                       -> no strength at all

The strength is located as the first *standalone* token that starts with a
number, so numbers glued to words in the name (``KENALOG-40``, ``24HR``,
``KLOR-CON M10``) stay part of the name.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

# --- vocabulary -------------------------------------------------------------

# Release / formulation tags that sit at the end of the name or start of the form.
RELEASE_TYPES = {"ER", "XR", "XL", "SR", "CR", "DR", "LA", "CD", "XT"}

# Spelling variants of strength units -> canonical unit.
STRENGTH_UNITS = {
    "MG": "MG",
    "MCG": "MCG",
    "GM": "GM",
    "GRAM": "GM",
    "G": "GM",
    "MEQ": "MEQ",
    "ML": "ML",
    "UNIT": "UNIT",
    "UNITS": "UNIT",
    "%": "%",
}

# Spelling variants of the "per" part of a concentration or rate.
PER_UNITS = {
    "ML": "ML",
    "GM": "GM",
    "GRAM": "GM",
    "HR": "HR",
    "HOUR": "HR",
    "DAY": "DAY",
    "ACT": "ACTUATION",
}

# Abbreviations used in the dosage-form part -> full word.
FORM_WORDS = {
    "TAB": "TABLET",
    "CAP": "CAPSULE",
    "CP": "CAPSULE",
    "SUSP": "SUSPENSION",
    "SOL": "SOLUTION",
    "SOLN": "SOLUTION",
    "CONC": "CONCENTRATE",
    "PTCH": "PATCH",
    "INHALR": "INHALER",
    "DRP": "DROPS",
    "DROP": "DROPS",
    "VL": "VIAL",
    "INJ": "INJECTION",
    "INJECT": "INJECTION",
    "INJCTR": "AUTO-INJECTOR",
    "AUTO-INJCT": "AUTO-INJECTOR",
    "AUTO-INJECTR": "AUTO-INJECTOR",
    "SUPP": "SUPPOSITORY",
    "SL": "SUBLINGUAL",
    "CHEW": "CHEWABLE",
    "DISPER": "DISPERSIBLE",
    "RESPUL": "RESPULE",
}

# Words that mark the start of the dosage form when a description has no strength.
FORM_START_WORDS = {
    "TABLET", "CAPSULE", "GEL", "PUMP", "EYE", "EAR", "DROPS", "OIL", "BODY",
    "SCALP", "ENEMA", "SOLUTION", "POWDER", "PACKET", "VAGINAL", "RING", "KIT",
    "NASAL", "SPRAY", "OINTMENT", "CREAM", "LOTION", "FOAM", "SUSPENSION",
}

# Ordered (first match wins): keyword in the dosage form -> broad category.
FORM_CATEGORIES = [
    ("Ophthalmic/Otic", {"EYE", "EAR", "EYELASH", "OPHTHALMIC", "OCUDOSE", "DROPS"}),
    ("Inhalation", {"INHALER", "DISKUS", "RESPULE"}),
    ("Nasal", {"NASAL"}),
    ("Injectable", {"VIAL", "SYRINGE", "PEN", "INJECTION", "AUTO-INJECTOR", "CARTRIDGES"}),
    ("Transdermal", {"PATCH"}),
    ("Vaginal/Rectal", {"VAGINAL", "RECTAL", "SUPPOSITORY", "ENEMA", "RING"}),
    ("Topical", {"CREAM", "GEL", "OINTMENT", "LOTION", "PUMP", "FOAM", "OIL", "EMULSION"}),
    ("Oral Liquid", {"SUSPENSION", "SOLUTION", "CONCENTRATE", "LIQUID", "SYRUP", "ELIXIR"}),
    ("Oral Solid", {"TABLET", "CAPSULE", "CAPLET", "PULVULE", "SOLUTAB", "INFATAB",
                    "TIMESPAN", "FILM", "DOSEPAK", "PACKET", "KIT"}),
]

# --- strength regex ----------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+"
_UNIT = "|".join(sorted(STRENGTH_UNITS, key=len, reverse=True))
_PER = "|".join(sorted(PER_UNITS, key=len, reverse=True))
_COMPONENT = rf"(?:{_NUM})\s*(?:{_UNIT})?"

STRENGTH_RE = re.compile(
    rf"""
    (?:^|(?<=\s))                                   # must start a token
    (?P<strength>
        {_COMPONENT}(?:\s*-\s*{_COMPONENT})*        # 10 MG | 5-325 MG | 75 MG-200 MCG
        (?:\s*/\s*(?P<per_qty>{_NUM})?\s*(?P<per_unit>{_PER}))?   # /5 ML | /24HR | /DAY
        (?:\s*\([^)]*\))?                           # TESTIM 1% (50 MG)
    )
    (?=\s|$)                                        # and end a token
    """,
    re.VERBOSE,
)
_COMPONENT_RE = re.compile(rf"(?P<num>{_NUM})\s*(?P<unit>{_UNIT})?")


@dataclass
class NdcParts:
    Drug_Name: str
    Release_Type: Optional[str] = None
    Strength: Optional[str] = None
    Strength_Value: Optional[float] = None
    Strength_Unit: Optional[str] = None
    Per_Quantity: Optional[float] = None
    Per_Unit: Optional[str] = None
    Ingredient_Count: Optional[int] = None
    Dosage_Form: Optional[str] = None
    Form_Category: str = "Unknown"


def _to_float(num: str) -> float:
    return float(num.replace(",", ""))


def _split_release(words: list[str], from_end: bool) -> tuple[list[str], Optional[str]]:
    """Pull a release tag (XR, DR, ...) off the end of the name or start of the form."""
    idx = -1 if from_end else 0
    if words and words[idx] in RELEASE_TYPES:
        tag = words[idx]
        return (words[:-1] if from_end else words[1:]), tag
    return words, None


def _normalise_form(words: list[str]) -> Optional[str]:
    if not words:
        return None
    return " ".join(FORM_WORDS.get(w, w) for w in words)


def _categorise_form(form: Optional[str], strength_unit: Optional[str]) -> str:
    """Map a dosage form to a broad category.

    "Unknown" means the description names no form; "Other" means it names one
    that isn't in ``FORM_CATEGORIES``.
    """
    if not form:
        return "Unknown"
    words = set(form.split())
    for category, keywords in FORM_CATEGORIES:
        if words & keywords:
            # "BETADINE 10% SOLUTION" is an antiseptic, not something you drink:
            # a %-strength solution/liquid with no "ORAL" is treated as topical.
            if category == "Oral Liquid" and strength_unit == "%" and "ORAL" not in words:
                return "Topical"
            return category
    return "Other"


def _parse_strength(text: str) -> tuple[Optional[float], Optional[str], int]:
    """Return (first value, its unit, number of ingredients) for a strength string.

    A component without a unit inherits the next unit that appears, so
    ``5-325 MG`` is 5 MG + 325 MG and ``0.3-0.4%`` is 0.3% + 0.4%.
    """
    main = text.split("(")[0].split("/")[0]
    components = [_COMPONENT_RE.match(part.strip()) for part in main.split("-")]
    components = [c for c in components if c]
    units = [c.group("unit") for c in components]
    for i in range(len(units) - 2, -1, -1):
        if units[i] is None:
            units[i] = units[i + 1]
    first_unit = STRENGTH_UNITS.get(units[0]) if units and units[0] else None
    return _to_float(components[0].group("num")), first_unit, len(components)


def parse_ndc_description(description: str) -> NdcParts:
    """Split one NDC description into name, release type, strength and form."""
    desc = re.sub(r"\s+", " ", str(description)).strip().upper()
    match, value, unit, n_ingredients = None, None, None, 0
    for candidate in STRENGTH_RE.finditer(desc):
        value, unit, n_ingredients = _parse_strength(candidate.group("strength"))
        # A lone number with no unit ("YAZ 28 TABLET") is a pack size, not a strength.
        # Unit-less combinations ("HYZAAR 100-12.5 TABLET") are still strengths.
        if unit is None and n_ingredients == 1 and not candidate.group("per_unit"):
            continue
        match = candidate
        break

    if match is None:
        # No strength: everything up to the first form word is the name.
        words = desc.split()
        cut = next((i for i, w in enumerate(words) if i > 0 and w in FORM_START_WORDS), len(words))
        name_words, release = _split_release(words[:cut], from_end=True)
        form = _normalise_form(words[cut:])
        return NdcParts(
            Drug_Name=" ".join(name_words),
            Release_Type=release,
            Dosage_Form=form,
            Form_Category=_categorise_form(form, None),
        )

    name_words, release = _split_release(desc[: match.start()].split(), from_end=True)
    form_words, form_release = _split_release(desc[match.end():].split(), from_end=False)
    strength = re.sub(r"\s*-\s*", "-", match.group("strength"))

    per_unit = match.group("per_unit")
    per_qty = match.group("per_qty")
    form = _normalise_form(form_words)

    return NdcParts(
        Drug_Name=" ".join(name_words),
        Release_Type=release or form_release,
        Strength=strength,
        Strength_Value=value,
        Strength_Unit=unit,
        Per_Quantity=(_to_float(per_qty) if per_qty else 1.0) if per_unit else None,
        Per_Unit=PER_UNITS[per_unit] if per_unit else None,
        Ingredient_Count=n_ingredients,
        Dosage_Form=form,
        Form_Category=_categorise_form(form, unit),
    )


def add_parsed_columns(df: pd.DataFrame, column: str = "NDC Description") -> pd.DataFrame:
    """Return a copy of ``df`` with the parsed description columns added."""
    unique = df[column].dropna().unique()
    parsed = pd.DataFrame(
        [asdict(parse_ndc_description(d)) for d in unique],
        index=pd.Index(unique, name=column),
    )
    parsed["Ingredient_Count"] = parsed["Ingredient_Count"].astype("Int64")
    out = df.drop(columns=[c for c in parsed.columns if c in df.columns])
    return out.join(parsed, on=column)
