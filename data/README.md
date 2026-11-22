# Input data

The clinical workbook is intentionally not distributed. Obtain authorized access independently before reproducing the clinical analysis. Use `python data/generate_example.py` for a synthetic workbook with matching column headers. The synthetic values are independent random scores and are not intended to reproduce clinical results.

The main program expects an Excel workbook with 132 consecutive numeric predictor columns, starting at `2 weeks::Biceps brachii_R` and ending at `2 weeks::AL_S4_5`. Motor columns are paired by `_R` and `_L`; sensory columns use `TR_`, `TL_`, `AR_`, `AL_` followed by the dermatome identifier. The target column is `6 months later::ASIAImpairmentScale`, coded 1–4 for AIS A–D. The main program maps 1–3 to 0 and 4 to 1. Column names retain the original database labels for compatibility; they are not a normative ISNCSCI terminology list.

The synthetic generator specifies the full ordered schema. No clinical records are used in generation. Do not commit patient data or output containing patient-level predictions.
