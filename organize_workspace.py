#!/usr/bin/env python3
"""
Organize the medica project workspace into a clean structure.
"""

import os
import shutil
from pathlib import Path


def organize_workspace():
    """
    Organize files into logical directories:
    - scripts/: Python scripts for processing
    - data/: Data files (JSON, JSONL)
    - reports/: Analysis reports (Markdown)
    - notebooks/: Jupyter notebooks
    - docs/: Documentation (README files)
    """
    
    print("="*70)
    print("ORGANIZING WORKSPACE")
    print("="*70)
    print()
    
    base_dir = Path.cwd()
    
    # Define directory structure
    directories = {
        'scripts': base_dir / 'scripts',
        'data': base_dir / 'data',
        'reports': base_dir / 'reports',
        'notebooks': base_dir / 'notebooks',
        'docs': base_dir / 'docs',
        'data/raw': base_dir / 'data' / 'raw',
        'data/processed': base_dir / 'data' / 'processed',
        'data/indexes': base_dir / 'data' / 'indexes',
    }
    
    # Create directories
    print("Creating directory structure...")
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {name}/")
    print()
    
    # File organization map
    file_moves = {
        # Scripts
        'analyze_captions.py': 'scripts/analyze_captions.py',
        'analyze_images.py': 'scripts/analyze_images.py',
        'analyze_images_comprehensive.py': 'scripts/analyze_images_comprehensive.py',
        'clean_captions.py': 'scripts/clean_captions.py',
        'compare_and_filter.py': 'scripts/compare_and_filter.py',
        'create_image_index.py': 'scripts/create_image_index.py',
        'create_training_index.py': 'scripts/create_training_index.py',
        'check_fig_exists.py': 'scripts/check_fig_exists.py',
        'check_fig_exists.sh': 'scripts/check_fig_exists.sh',
        'filter_json_keys.py': 'scripts/filter_json_keys.py',
        'move_matching_files.py': 'scripts/move_matching_files.py',
        'process_json_pipeline.py': 'scripts/process_json_pipeline.py',
        'remove_caption_key.py': 'scripts/remove_caption_key.py',
        'remove_null_captions.py': 'scripts/remove_null_captions.py',
        'remove_null_references.py': 'scripts/remove_null_references.py',
        
        # Raw data
        's2_full_figures.jsonl': 'data/raw/s2_full_figures.jsonl',
        's2_parsed.json': 'data/raw/s2_parsed.json',
        's2_parsed_filtered.json': 'data/raw/s2_parsed_filtered.json',
        's2_final.json': 'data/raw/s2_final.json',
        'pipa.json': 'data/raw/pipa.json',
        'pipe.json': 'data/raw/pipe.json',
        'test.json': 'data/raw/test.json',
        'test_first5.json': 'data/raw/test_first5.json',
        
        # Processed data
        's2_cleaned.json': 'data/processed/s2_cleaned.json',
        's2_cleaned_missing_images.json': 'data/processed/s2_cleaned_missing_images.json',
        'comparison_summary.json': 'data/processed/comparison_summary.json',
        'stats.txt': 'data/processed/stats.txt',
        
        # Indexes
        'image_index.json': 'data/indexes/image_index.json',
        
        # Reports
        's2_final_analysis_report.md': 'reports/s2_final_analysis_report.md',
        's2_cleaned_analysis_report.md': 'reports/s2_cleaned_analysis_report.md',
        's2_cleaned_image_analysis.md': 'reports/s2_cleaned_image_analysis.md',
        'filtered_figures_image_analysis.md': 'reports/filtered_figures_image_analysis.md',
        
        # Notebooks
        'model_training_preprocessing.ipynb': 'notebooks/model_training_preprocessing.ipynb',
        'model_usage.ipynb': 'notebooks/model_usage.ipynb',
        
        # Documentation
        'README_CAPTION_ANALYSIS.md': 'docs/README_CAPTION_ANALYSIS.md',
        'README_INSTRUCTIONS.md': 'docs/README_INSTRUCTIONS.md',
        'README_PROGRESS.md': 'docs/README_PROGRESS.md',
    }
    
    # Move files
    print("Moving files...")
    moved_count = 0
    skipped_count = 0
    
    for source, destination in file_moves.items():
        source_path = base_dir / source
        dest_path = base_dir / destination
        
        if source_path.exists():
            try:
                shutil.move(str(source_path), str(dest_path))
                print(f"  ✓ {source} → {destination}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ Failed to move {source}: {e}")
        else:
            skipped_count += 1
    
    print()
    print(f"✓ Moved {moved_count} files")
    if skipped_count > 0:
        print(f"  (Skipped {skipped_count} files - not found)")
    print()
    
    # Create a README in root
    print("Creating root README.md...")
    readme_content = """# Medical Image Caption Dataset

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
"""
    
    readme_path = base_dir / 'README.md'
    if not readme_path.exists() or readme_path.name == 'readme.md':
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✓ Created {readme_path}")
    else:
        print(f"  README.md already exists, skipping")
    print()
    
    # Create .gitignore if not exists
    gitignore_path = base_dir / '.gitignore'
    if not gitignore_path.exists():
        print("Creating .gitignore...")
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files
*.json
!requirements.json
!package.json
*.jsonl
*.csv
*.tsv
*.txt
!requirements.txt

# Images
*.png
*.jpg
*.jpeg
*.tiff
*.tif
*.dcm

# Reports (optional - comment out if you want to commit reports)
# reports/*.md

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large datasets
dataset/images/*
data/raw/*
data/processed/*.json
data/indexes/*.json

# Keep structure
!dataset/.gitkeep
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/indexes/.gitkeep
"""
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print(f"✓ Created {gitignore_path}")
    print()
    
    print("="*70)
    print("WORKSPACE ORGANIZED!")
    print("="*70)
    print()
    print("New structure:")
    print("  scripts/       - All Python processing scripts")
    print("  data/raw/      - Original unprocessed data")
    print("  data/processed/ - Cleaned and filtered data")
    print("  data/indexes/  - Lookup indexes")
    print("  reports/       - Analysis reports (.md)")
    print("  notebooks/     - Jupyter notebooks")
    print("  docs/          - Documentation")
    print()
    print("✓ Root README.md created with project overview")
    print("✓ .gitignore created to exclude large files")
    print()
    print("You can now commit the organized structure to git!")
    print()


if __name__ == "__main__":
    organize_workspace()
