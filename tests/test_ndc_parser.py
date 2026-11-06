from pathlib import Path

import pandas as pd
import pytest

from drug_price_optimization.data.ndc_parser import add_parsed_columns, parse_ndc_description

DATA = Path(__file__).parents[1] / "data/raw/cleaned_nadac-national-average-drug-acquisition-cost-05-14-2025.csv"

# (description, expected fields) — all taken from the real NADAC file.
CASES = [
    ("ABILIFY 10 MG TABLET",
     dict(Drug_Name="ABILIFY", Strength="10 MG", Strength_Value=10.0, Strength_Unit="MG",
          Ingredient_Count=1, Dosage_Form="TABLET", Form_Category="Oral Solid")),
    ("FOCALIN XR 20 MG CAPSULE",
     dict(Drug_Name="FOCALIN", Release_Type="XR", Strength_Value=20.0, Dosage_Form="CAPSULE")),
    ("UCERIS 9 MG ER TABLET",
     dict(Drug_Name="UCERIS", Release_Type="ER", Dosage_Form="TABLET")),
    ("PERCOCET 5-325 MG TABLET",
     dict(Drug_Name="PERCOCET", Strength="5-325 MG", Strength_Value=5.0, Strength_Unit="MG", Ingredient_Count=2)),
    ("TRIBENZOR 40-5-12.5 MG TABLET",
     dict(Strength="40-5-12.5 MG", Ingredient_Count=3)),
    ("ARTHROTEC 75 MG-200 MCG TAB",
     dict(Strength="75 MG-200 MCG", Strength_Unit="MG", Ingredient_Count=2, Dosage_Form="TABLET")),
    ("KAZANO 12.5-1,000 MG TABLET",
     dict(Strength="12.5-1,000 MG", Strength_Value=12.5, Ingredient_Count=2)),
    ("CANASA 1,000 MG SUPPOSITORY",
     dict(Strength_Value=1000.0, Form_Category="Vaginal/Rectal")),
    ("LOTEMAX 0.5% EYE DROPS",
     dict(Drug_Name="LOTEMAX", Strength="0.5%", Strength_Value=0.5, Strength_Unit="%",
          Dosage_Form="EYE DROPS", Form_Category="Ophthalmic/Otic")),
    ("COMBIGAN 0.2%-0.5% EYE DROPS",
     dict(Strength_Unit="%", Ingredient_Count=2)),
    ("SYSTANE 0.3-0.4% EYE DROP",
     dict(Strength_Unit="%", Ingredient_Count=2, Dosage_Form="EYE DROPS")),
    ("BETADINE 10% SOLUTION",
     dict(Form_Category="Topical")),
    ("ZOLOFT 20 MG/ML ORAL CONC",
     dict(Strength="20 MG/ML", Per_Quantity=1.0, Per_Unit="ML",
          Dosage_Form="ORAL CONCENTRATE", Form_Category="Oral Liquid")),
    ("ERYPED 200 MG/5 ML SUSPENSION",
     dict(Strength_Value=200.0, Per_Quantity=5.0, Per_Unit="ML")),
    ("BUTRANS 10 MCG/HR PATCH",
     dict(Strength_Unit="MCG", Per_Quantity=1.0, Per_Unit="HR", Form_Category="Transdermal")),
    ("EXELON 4.6 MG/24HR PATCH",
     dict(Per_Quantity=24.0, Per_Unit="HR")),
    ("DAYTRANA 20 MG/9 HOUR PATCH",
     dict(Per_Quantity=9.0, Per_Unit="HR")),
    ("TRANSDERM-SCOP 1 MG/3 DAY PTCH",
     dict(Drug_Name="TRANSDERM-SCOP", Per_Quantity=3.0, Per_Unit="DAY", Dosage_Form="PATCH")),
    ("APRISO ER 0.375 GRAM CAPSULE",
     dict(Release_Type="ER", Strength_Unit="GM")),
    ("KLOR-CON 10 MEQ TABLET",
     dict(Drug_Name="KLOR-CON", Strength_Unit="MEQ")),
    ("TESTIM 1% (50 MG) GEL",
     dict(Strength="1% (50 MG)", Strength_Value=1.0, Strength_Unit="%", Ingredient_Count=1, Dosage_Form="GEL")),
    # Numbers glued to the name stay in the name.
    ("KENALOG-40 40 MG/ML VIAL",
     dict(Drug_Name="KENALOG-40", Strength_Value=40.0, Form_Category="Injectable")),
    ("PREVACID 24HR DR 15 MG CAPSULE",
     dict(Drug_Name="PREVACID 24HR", Release_Type="DR", Strength_Value=15.0)),
    ("KLOR-CON M10 TABLET",
     dict(Drug_Name="KLOR-CON M10", Strength=None, Dosage_Form="TABLET")),
    # Unit-less combination is a strength; unit-less single number is a pack size.
    ("HYZAAR 100-12.5 TABLET",
     dict(Drug_Name="HYZAAR", Strength="100-12.5", Strength_Unit=None, Ingredient_Count=2)),
    ("YAZ 28 TABLET",
     dict(Drug_Name="YAZ 28", Strength=None, Dosage_Form="TABLET")),
    # No strength at all.
    ("FLEET ENEMA",
     dict(Drug_Name="FLEET", Strength=None, Dosage_Form="ENEMA", Form_Category="Vaginal/Rectal")),
    ("COSOPT PF EYE DROPS",
     dict(Drug_Name="COSOPT PF", Dosage_Form="EYE DROPS")),
    # No dosage form at all.
    ("DEPO-TESTOSTERONE 200 MG/ML",
     dict(Drug_Name="DEPO-TESTOSTERONE", Dosage_Form=None, Form_Category="Unknown")),
]


@pytest.mark.parametrize("description, expected", CASES, ids=[c[0] for c in CASES])
def test_parse_ndc_description(description, expected):
    parts = parse_ndc_description(description)
    for field, value in expected.items():
        assert getattr(parts, field) == value, field


def test_whitespace_and_case_are_normalised():
    assert parse_ndc_description("  abilify   10 mg  tablet ").Drug_Name == "ABILIFY"


def test_add_parsed_columns_keeps_rows_and_adds_columns():
    df = pd.DataFrame({"NDC Description": ["ABILIFY 10 MG TABLET", "ABILIFY 10 MG TABLET", None]})
    out = add_parsed_columns(df)
    assert len(out) == 3
    assert list(out["Drug_Name"][:2]) == ["ABILIFY", "ABILIFY"]
    assert out["Strength_Value"].iloc[0] == 10.0


@pytest.mark.skipif(not DATA.exists(), reason="NADAC CSV not present")
def test_every_real_description_gets_a_name_and_category():
    df = pd.read_csv(DATA, usecols=["NDC Description"]).drop_duplicates()
    out = add_parsed_columns(df)
    assert out["Drug_Name"].str.len().gt(0).all()
    assert not out["Drug_Name"].str.contains(r"\b\d+(?:\.\d+)?\s*(?:MG|MCG|%)").any()
    assert (out["Form_Category"] == "Other").sum() == 0
