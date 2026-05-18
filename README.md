# ML-based compression algorithm selection

Bachelor's thesis project that was made for a purpose to predict the best
compression algorithm for a given file using classification on mostly
byte-level features.

## Quick start

```bash
pip install -r requirements.txt

python src/a0_clean.py
python src/a1_registry.py
python src/a2_benchmark.py --max-size 100000000 --resume
python src/a3_features.py
python src/a4_validate.py
python src/a5_prepare.py
python src/b1_train.py
python src/b2_threshold.py
python src/b3_test.py
```

## Pipeline

| Stage | Script | Purpose |
|-------|--------|---------|
| A0 | `src/a0_clean.py` | Deduplicate raw files by SHA256 |
| A1 | `src/a1_registry.py` | Build file registry (size, type group) |
| A2 | `src/a2_benchmark.py` | Run 20 compressors on each file |
| A3 | `src/a3_features.py` | Extract 263 features per file |
| A4 | `src/a4_validate.py` | Integrity checks across CSVs |
| A5 | `src/a5_prepare.py` | Stratified 70/15/15 train/val/test split |
| B1 | `src/b1_train.py` | Train 4 models × 5 profiles |
| B2 | `src/b2_threshold.py` | Tune confidence threshold on validation |
| B3 | `src/b3_test.py` | Evaluate on test set with Wilcoxon test |
