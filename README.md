# Medical Image Caption Dataset

A curated dataset of medical images with cleaned captions for machine learning training.

## 📁 Project Structure

```
medica/
├── scripts/           # Python scripts for data processing
├── data/
│   ├── raw/          # Original unprocessed data
│   ├── processed/    # Cleaned and processed data
│   └── indexes/      # Image and training indexes
├── reports/          # Analysis reports and statistics
├── notebooks/        # Jupyter notebooks for experimentation
├── docs/             # Documentation and progress tracking
├── dataset/          # Dataset files (images, captions, metadata)
└── venv/             # Python virtual environment
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Processing Pipeline
```bash
# Clean captions
python3 scripts/clean_captions.py data/raw/s2_final.json

# Analyze cleaned data
python3 scripts/analyze_captions.py data/processed/s2_cleaned.json

# Create image index (O(1) lookup)
python3 scripts/create_image_index.py /path/to/images

# Compare captions with images
python3 scripts/compare_and_filter.py data/processed/s2_cleaned.json

# Create training index
python3 scripts/create_training_index.py data/processed/s2_cleaned_with_images.json /path/to/images
```

## 📊 Dataset Statistics

- **Total Captions (Original):** 138,202
- **Cleaned Captions:** 105,723 (76.5% retention)
- **Images Available:** See `data/processed/comparison_summary.json`
- **Average Caption Length:** 46 words (median)
- **Medical Term Coverage:** 88.1%

## 📝 Documentation

- **Progress Tracking:** `docs/README_PROGRESS.md`
- **Caption Analysis:** `docs/README_CAPTION_ANALYSIS.md`
- **Instructions:** `docs/README_INSTRUCTIONS.md`

## 📈 Reports

All analysis reports are in `reports/`:
- Pre-cleaning analysis: `s2_final_analysis_report.md`
- Post-cleaning analysis: `s2_cleaned_analysis_report.md`
- Image analysis: `filtered_figures_image_analysis.md`

## 🔧 Scripts

### Data Processing
- `clean_captions.py` - Clean and normalize captions
- `analyze_captions.py` - Comprehensive caption analysis
- `analyze_images_comprehensive.py` - Detailed image analysis

### Index Creation
- `create_image_index.py` - O(1) image lookup index
- `create_training_index.py` - Training-ready dataset index
- `compare_and_filter.py` - Match captions with images

### Utilities
- `filter_json_keys.py` - Filter JSON by keys
- `remove_null_*.py` - Clean null entries
- `process_json_pipeline.py` - Full processing pipeline

## 📦 Data Files

### Raw Data (`data/raw/`)
- `s2_full_figures.jsonl` - Original S2ORC data
- `s2_final.json` - Pre-processed data

### Processed Data (`data/processed/`)
- `s2_cleaned.json` - Cleaned captions (105,723 entries)
- `s2_cleaned_missing_images.json` - Entries without images
- `comparison_summary.json` - Caption-image matching statistics

### Indexes (`data/indexes/`)
- `image_index.json` - O(1) image lookup by pdf_hash

## 📄 License

See `LICENSE`, `DATA_LICENSE.txt`, and `LICENSE-THESIS.txt` for details.

## 🔬 Next Steps

1. ✅ Caption cleaning and normalization
2. ✅ Image-caption matching
3. ✅ Index creation for O(1) lookup
4. ⏳ Image preprocessing (resize, normalize)
5. ⏳ Train/validation/test split (80/10/10)
6. ⏳ Model training

---

*Last updated: October 15, 2025*
