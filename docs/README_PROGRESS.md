# Project Progress Log

This log tracks key actions, decisions, and artifacts added to the repository over time. New entries are appended with timestamps.

Format:
- Date (YYYY-MM-DD)
- Change summary
- Files affected
- Notes/next steps

---

## 2025-10-14

Change:
- Added a robust streaming extractor script for large JSONL files that selects the first input-like field and writes it to a new file.

Files:
- `extract_first_input_field.py`

Notes:
- Memory-safe (line-by-line), supports `.gz`, configurable field selection, outputs values or JSONL. Did not attempt to read large files in this environment.

---

## 2025-10-14

Change:
- Added a simple script to keep the first N keys (default 5) from a single JSON object and write the result to a new file.

Files:
- `keep_first_n_keys.py`

Notes:
- Preserves key order; validates input is a JSON object; configurable N.

Update:
- Adjusted to run with built-in defaults (no CLI required): reads `test.json`, writes `test_first5.json`, keeps first 5 keys.

---

## 2025-10-14

Change:
- Modified the trimming script to keep a specific set of keys by default: `fig_key`, `fig_uri`, `s2_caption`, `s2orc_caption`.

Files:
- `keep_first_n_keys.py`

Notes:
- Uses DEFAULT_KEYS to control which fields are kept (missing keys are ignored). Still runs without CLI.

Update:
- Simplified the script to bare-bones Python: no functions, no type hints, no error handling—just read, filter, write.
- Renamed to `filter_json_keys.py` for clarity.
- Added `pdf_hash` to keys; now combines `pdf_hash`, `fig_key`, and `fig_uri` into a new `combined_id` field using underscore separator.
- Final output now contains only 3 fields: `fig_id` (combination of pdf_hash and fig_uri), `s2_caption`, and `s2orc_caption`.
- Changed to keep only `pdf_hash`, `s2orc_caption`, and `s2orc_references` (no combination, no fig_id).
- Fixed to handle JSONL format: reads line-by-line, processes each JSON object, outputs to JSONL (s2_parsed.jsonl).

---

## 2025-10-14

Change:
- Added a script to remove entries with null `s2orc_caption` from JSONL files.

Files:
- `remove_null_captions.py`

Notes:
- Reads `s2_parsed.jsonl`, filters out null captions, writes to `s2_parsed_filtered.jsonl`. Prints summary stats.

---

## 2025-10-14

Change:
- Added scripts to handle null values in parsed data.

Files:
- `remove_caption_key.py` - Removes the `s2orc_caption` key itself when null (keeps entry).
- `remove_null_references.py` - Removes entire entries where `s2orc_references` is null.

Notes:
- `remove_caption_key.py`: input `s2_parsed_filtered.jsonl`, output `s2_parsed_no_null_keys.jsonl`
- `remove_null_references.py`: input `s2_parsed_filtered.jsonl`, output `s2_parsed_with_refs.jsonl`

Update:
- Combined all filtering scripts into one pipeline: `process_json_pipeline.py`
- Single script runs all 3 steps: extract keys, remove null captions, remove null references
- Provides detailed progress output and summary stats
- Modified to combine `pdf_hash` and `fig_uri` with underscore separator: `pdf_hash_fig_uri`

---

## 2025-10-14

Change:
- Added a bash script to check if a file matching `fig_id` exists in a specified folder.

Files:
- `check_fig_exists.sh`

Notes:
- Reads `fig_id` from JSON file, loops through folder, reports if file found or not found.
- Reverted to handle single JSON object (not JSONL).

Update:
- Converted bash script to Python: `check_fig_exists.py` - same logic, cleaner syntax.
- Updated to read `s2_final.json` (JSONL format), process all entries, check each `pdf_hash` against image folder, and provide summary stats.
- Optimized for speed: preloads directory listing into a set for O(1) lookups instead of scanning folder for every entry. Dramatically faster on large datasets.
- Ultra-optimized: frozenset (faster read-only), larger file buffer (65KB), progress updates, batch missing output, pre-compiled method lookups, short-circuit evaluation.

---

## 2025-10-14

Change:
- Added script to move files matching JSON entries from source to destination folder.

Files:
- `move_matching_files.py`

Notes:
- Reads `s2_final.json`, finds matching files by `pdf_hash`, moves them to `filtered_figures` folder.
- Includes dry-run mode (set `dry_run = True` to preview without moving).
- Progress updates every 10,000 entries, error handling, summary stats.

---

## 2025-10-14

Change:
- Created simple caption analyzer for the filtered dataset.

Files:
- `analyze_captions.py`

Notes:
- Analyzes `s2_final.json` with no external dependencies (pure Python).
- Provides: basic stats, medical term analysis, figure type classification, quality checks, sample captions.
- Tracks modalities (X-ray, MRI, CT, ultrasound, microscopy, charts), anatomy terms, and findings.

---

## 2025-10-14

Change:
- Created comprehensive caption cleaning script with configurable rules.

Files:
- `clean_captions.py`

Notes:
- Input: `s2_final.json`, Output: `s2_cleaned.json`
- Configurable rules: remove short/long captions, remove figure prefixes, filter by medical terms, remove generic phrases, remove duplicates, clean whitespace, filter by modality
- Provides detailed statistics on what was cleaned/removed
- All rules can be enabled/disabled at the top of the file

Update:
- Optimized based on analysis of 138K captions: disabled very_long filter (keep detailed captions), updated medical terms to match actual data (CT, MRI, cell, lesion, mass), adjusted threshold to 200 words.
- Added punctuation cleaning: fixes multiple periods, commas, spacing issues around punctuation.
- Added repeated words fix: detects and fixes patterns like "the the" → "the" (affects 0.5% - 716 captions).
- Enhanced with 16 comprehensive cleaning rules organized by priority:
  * CRITICAL FIXES: remove figure prefixes (56.9%), incomplete sentences (22.7%), very short captions (0.7%), URLs (1.0%), citations (0.8%), repeated words (0.5%)
  * QUALITY IMPROVEMENTS: clean whitespace, punctuation, capitalization, special chars, brackets
  * OPTIONAL FILTERS: configurable removal of very long, non-medical, generic, duplicate captions
- Script now resolves all critical data quality issues identified in analysis report.

---

## 2025-10-15

Change:
- Enhanced caption analyzer with comprehensive metrics and Markdown report generation.

Files:
- `analyze_captions.py` (updated)

Notes:
- Added command-line argument support: `python3 analyze_captions.py [input_file]`
- Auto-generates output filename: `<input_file>_analysis_report.md`
- Generates detailed Markdown reports with ASCII/Unicode diagrams (horizontal bar charts, histograms)
- Enhanced analysis sections:
  * Basic statistics with standard deviation, percentiles, distribution histograms
  * Punctuation analysis (8 patterns)
  * Quality indicators (good/descriptive/technical/quantitative language)
  * Problematic patterns detection (citations, URLs, excessive caps, repeated words)
  * Most common words and bigram analysis
  * Reference statistics with distribution
  * Length distribution by bins
  * Enhanced quality checks (9 metrics)
  * Data completeness report
- Analyzed s2_final.json (138,202 captions) - report saved as s2_final_analysis_report.md
- Key findings: 56.9% start with "Figure", 22.7% incomplete sentences, 0.7% very short, 88.4% contain medical terms

---

## 2025-10-13

Change:
- Added a dedicated guide for S2ORC-specific deep caption analysis (Path B) as a code-free checklist.

Files:
- `README_CAPTION_ANALYSIS.md`

Notes:
- Covers scope, steps 1–10, edge cases, optional add-ons, and a summary checklist.

---

Earlier work (summary):
- Set up a comprehensive `.gitignore`.
- Wrote `README_INSTRUCTIONS.md` with a project roadmap.
- Implemented analysis tools: `caption_analysis_tools.py`, `quick_analysis.py`, `caption_explorer.py`, and `s2_caption_analyzer.py` (with None-safe handling for S2ORC fields).
