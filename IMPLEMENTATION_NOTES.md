# Implementation Summary: Image Comparison & Markdown Rewriting

## Overview
Implemented comprehensive image matching and markdown rewriting system with three-tier comparison strategy: MD5, perceptual hash (phash), and Ollama-based visual comparison.

## Changes Made

### 1. **pdf2foundry/import-pdf.py** – Complete Refactor
- **Added imports**: `hashlib`, `Tuple`
- **Optional dependencies** (graceful fallback if missing):
  - `PIL` (Pillow) for image handling
  - `imagehash` for perceptual hash computation
  - `ollama` for LLM-based image comparison

#### New Helper Functions:

- **`md5_file(path: str) -> str`**
  - Computes MD5 hash of a file for exact matching
  - Reads file in chunks to handle large files efficiently

- **`phash_distance(a: str, b: str, logger=None) -> int`**
  - Computes perceptual hash distance between two images
  - Returns 9999 if PIL/imagehash unavailable (graceful fallback)
  - Logs debug info on comparison failure

- **`ollama_compare_images(img_a: str, img_b: str, logger=None) -> bool`**
  - Uses `ollama` library with `llava:34b` model for visual comparison
  - Converts images to base64 for API transmission
  - Parses model response to determine similarity
  - Returns `False` if ollama unavailable or request fails

- **`rewrite_markdown_image_references(md_text, output_dir, images_dir, png_paths, logger) -> str`**
  - **Moved from inline code** – now reusable helper function
  - **Three-tier matching strategy**:
    1. Exact MD5 match (fastest)
    2. Perceptual hash match (if PIL/imagehash available, distance threshold ≤ 5)
    3. Ollama visual comparison (if ollama available)
  - **File organization**:
    - Matched images: original moved to `output_dir/images_replaced/` with `{png_basename}_replaced{orig_ext}` naming
    - Unmatched images: moved to `output_dir/images/`
  - **Markdown updates**:
    - Replaces `![alt](original.jpg)` with `![alt](images/extracted001.png)` when matched
    - Replaces `<img src="...">` similarly
    - Handles relative paths and `./` prefixes

#### Updated `convert_pdf_to_markdown()`
- Now uses `rewrite_markdown_image_references()` helper
- Cleaner control flow with less inline logic

### 2. **tests/test_import_pdf.py** – New Unit Test Suite
Comprehensive test coverage (11 test cases across 5 test classes):

#### `TestMD5File`
- `test_md5_file_consistency`: Verifies stable hashing
- `test_md5_file_different_content`: Different files produce different hashes
- `test_md5_file_nonexistent`: Error handling for missing files

#### `TestPhashDistance`
- `test_phash_distance_none_when_pil_missing`: Returns 9999 if PIL unavailable
- `test_phash_distance_none_when_imagehash_missing`: Returns 9999 if imagehash unavailable

#### `TestOllamaCompareImages`
- `test_ollama_compare_images_none_when_ollama_missing`: Returns False if ollama unavailable
- `test_ollama_compare_images_returns_bool`: Returns boolean type
- `test_ollama_compare_images_false_on_no`: Correctly interprets "no" response

#### `TestRewriteMarkdownImageReferences`
- `test_rewrite_markdown_md_image_reference`: Finds and rewrites Markdown image refs
- `test_rewrite_markdown_html_image_reference`: Finds and rewrites HTML image refs
- `test_rewrite_markdown_images_replaced_folder_created`: Creates `images_replaced/` directory
- `test_rewrite_markdown_original_file_moved_with_suffix`: Renames moved files with `_replaced` suffix
- `test_rewrite_markdown_no_match_moves_to_images`: Unmatched images move to `images/`
- `test_rewrite_markdown_normalizes_paths`: Handles relative paths (e.g., `./image.jpg`)
- `test_rewrite_markdown_multiple_references`: Processes multiple image refs in single markdown

## Key Features

### Image Matching Strategy
1. **MD5 Exact Match**: Fast, reliable for identical files
2. **Perceptual Hash (phash)**: Matches visually similar images despite compression/format changes
   - Threshold: distance ≤ 5 (tunable in code)
3. **Ollama Visual Comparison**: AI-based similarity check using `llava:34b` model
   - Requires locally running Ollama instance
   - Uses base64-encoded images

### File Organization
```
output_dir/
├── {pdf_name}/
│   └── {pdf_name}.md
├── images/
│   ├── extracted001.png
│   ├── extracted002.png
│   └── unmatched_image.jpg
└── images_replaced/
    ├── extracted001_replaced.jpg  (original matched image)
    └── extracted002_replaced.png
```

### Error Handling & Logging
- All subprocess calls wrapped with `try/except` + logging
- Optional dependencies fail gracefully (warnings, not errors)
- Detailed logging at INFO/DEBUG levels
- User-friendly error messages to stdout

## Dependencies

### Required (in setup/install_python.sh)
- `marker-pdf[all]`
- `pytesseract`
- `psutil`
- `poppler-utils`
- `ollama`

### Optional (improve matching)
- `pillow` (PIL) – for image processing
- `imagehash` – for perceptual hashing

## Running Tests

```bash
# From repository root
cd tests

# Run all tests
python -m unittest test_import_pdf

# Run specific test class
python -m unittest test_import_pdf.TestRewriteMarkdownImageReferences

# Run specific test
python -m unittest test_import_pdf.TestMD5File.test_md5_file_consistency

# Verbose output
python -m unittest test_import_pdf -v
```

## Usage Example

```python
from pdf2foundry.import_pdf import (
    rewrite_markdown_image_references,
    ollama_compare_images,
    md5_file,
)
import logging

logger = logging.getLogger("my_app")

# Rewrite image refs with custom matching
md_text = "![original](doc_image.jpg)"
png_paths = ["/output/images/page-001.png"]
result = rewrite_markdown_image_references(
    md_text, "/output", "/output/images", png_paths, logger
)
# Returns: "![original](images/page-001.png)"
# Moves doc_image.jpg to /output/images_replaced/page-001_replaced.jpg
```

## Configuration & Tuning

### Perceptual Hash Threshold
Edit in `rewrite_markdown_image_references()`:
```python
if best is not None and best_dist <= 5:  # Tune here (lower = stricter)
    matched_png = best
```

### Ollama Model
Edit in `ollama_compare_images()`:
```python
response = ollama.generate(
    model="llava:34b",  # Change model here
    prompt=prompt,
    images=[img_a_b64, img_b_b64],
    stream=False,
)
```

## Future Enhancements
- Batch processing with progress tracking
- Parallel image comparison for large documents
- Configurable matching strategy priority
- Cache Ollama responses to avoid redundant calls
- Integration with Flask API for web-based processing
