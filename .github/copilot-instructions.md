# pdf2foundry AI Coding Instructions

## Project Overview
pdf2foundry converts TTRPG (tabletop RPG) PDFs into Foundry VTT modules with extracted content organized into Compendiums. The project uses Marker (LLM-based PDF processor) for markdown conversion, pdfimages for image extraction, and multi-tier image matching to handle content reuse.

## Architecture & Key Components

### Core Processing Pipeline (import_pdf.py)
The main workflow orchestrates three external tools with intelligent fallback handling:
1. **Marker CLI** (`run_marker_single`) - Converts PDF to markdown using local LLM (via Ollama)
2. **pdfimages CLI** (`run_pdfimages_into_dir`) - Extracts images from PDF as PNG files
3. **Image Matching** (`find_matching_png`) - Maps extracted images to markdown references using three-tier strategy

### Image Matching Strategy
Implements graceful degradation with three matching tiers in `find_matching_png()`:
- **Tier 1: MD5 Exact Match** - Fast exact file comparison using `md5_file()`
- **Tier 2: Perceptual Hash** - Visual similarity detection via `phash_distance()` (threshold ≤ 5)
  - Gracefully returns 9999 if PIL/imagehash unavailable
- **Tier 3: Ollama Visual Comparison** - Uses `llava:34b` model via `ollama_compare_images()`
  - Requires local Ollama instance running
  - Falls back to False if unavailable

### File Organization Pattern
```
output_dir/
├── {pdf_name}/
│   └── {pdf_name}.md          # Main markdown output from Marker
│   └── images/
│       └── extracted001.png       # Unmatched images (keep originals)
│   └── images_replaced/
        └── extracted001_replaced.jpg  # Matched images (originals with _replaced suffix)
```

## Development Patterns

### Error Handling & Logging
- Use `logging.Logger` parameter throughout (passed from CLI)
- Gracefully degrade when optional dependencies missing (PIL, imagehash, ollama)
- Return sentinel values on failures:
  - `phash_distance()` returns `9999` (indicates failure)
  - `ollama_compare_images()` returns `False` (no match assumed)
  - CLI commands return empty lists or None on subprocess failures
- Always log at appropriate levels: `info` for workflow, `debug` for details, `warning` for fallbacks, `error` for failures

### Subprocess Integration Pattern
Follow the pattern in `run_marker_single()` and `run_pdfimages_into_dir()`:
- Use `subprocess.run()` with `capture_output=True, text=True, check=True`
- Catch `CalledProcessError` and log stderr/stdout
- Return sensible defaults on failure (None or empty list)
- Never raise exceptions - let caller decide how to handle

### Image Path Handling
- Normalize paths with `os.path.normpath()` before comparison
- Handle both absolute and relative paths (e.g., `./image.jpg`)
- Use `os.path.basename()` to extract filenames for comparison

## Type Safety & Code Standards

### Mypy Configuration
Project enforces strict typing:
- `disallow_untyped_defs = true` - All functions must have type hints
- `disallow_untyped_calls = true` - All function calls must be typed
- `disallow_any_generics = true` - No bare `List`, use `List[T]`
- Exception: `[mypy-fire]` excludes fire CLI framework

**Pattern:** Always include full type signatures:
```python
def process_pdf(pdf_path: str, output_dir: str, logger: logging.Logger) -> Optional[str]:
    """Docstring describing purpose."""
```

### Code Formatting
- **Black**: Line length 88 (configured in .flake8)
- **isort**: Import sorting (configured in .isort.cfg)
- **Flake8**: RST docstring convention, max complexity 18
  - Ignores: E203, E266, E501, W503, F403, F401
- Run all formatters via: `tox -e lint` or individually:
  ```bash
  poetry run isort pdf2foundry
  poetry run black pdf2foundry tests
  poetry run flake8 pdf2foundry
  poetry run mypy -m pdf2foundry --exclude ^tests
  ```

## Testing Approach

### Test Organization (tests/test_import_pdf.py)
Group tests by function using unittest.TestCase subclasses:
- Mock external dependencies (PIL, imagehash, ollama) since not always available
- Use `tempfile.mkdtemp()` for file operations - always clean up in `tearDown()`
- Test both success and failure paths (graceful degradation)

**Example pattern:**
```python
class TestPhashDistance(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test")
    
    def test_error_handling(self):
        # Test with invalid files returns 9999
        distance = phash_distance("invalid.png", "another.png", self.logger)
        self.assertEqual(distance, 9999)
```

### Running Tests
```bash
# Run all tests with coverage
poetry run pytest -s --cov=pdf2foundry tests

# Run via tox (tests all Python versions 3.8-3.11)
tox

# Run specific test
poetry run pytest tests/test_import_pdf.py::TestMD5File::test_md5_file_consistency
```

## External Dependencies & Integration Points

### Required System Tools
- **marker_single CLI** - From marker package, used for PDF→Markdown conversion with LLM
- **pdfimages CLI** - From poppler-utils, extracts images from PDF
- **Local Ollama** - For image comparison and LLM processing (optional but recommended)

### Python Dependencies
- **fire** (0.4.0) - CLI framework for module commands
- **Optional for image processing:**
  - PIL/Pillow - Image loading and manipulation
  - imagehash - Perceptual hashing
  - ollama - Local model integration

### Model Configuration (models.py)
Project defines two cost-optimized tiers (aspirational, not yet fully integrated):
- **demo_tier**: Fast models for 50k CCU (mistral:7b, moondream2)
- **full_tier**: Better quality for paid users (llama3.3:70b, llava:34b)

## Versioning & Dependencies

### Python Support
Supports Python 3.12

### Poetry-Based Dependency Management
- Core: fire (0.4.0)
- Optional groups: test, dev, doc
- Install dev environment: `poetry install -E test -E doc -E dev`

### Pre-commit Hooks
Project uses pre-commit for linting/formatting - configuration in `.pre-commit-config.yaml`

## Common Tasks

### Add New Image Matching Strategy
1. Create new matching function in `import_pdf.py` following pattern: `def new_matcher(img_a: str, img_b: str, logger: logging.Logger) -> Union[bool, int]:`
2. Add to `find_matching_png()` after existing tiers with appropriate threshold
3. Handle missing dependencies gracefully (return sentinel value)
4. Add unit tests in `TestRewriteMarkdownImageReferences` or new test class
5. Update IMPLEMENTATION_NOTES.md with strategy details

### Extend Markdown Rewriting
Edit `rewrite_markdown_image_references()` in `import_pdf.py`:
- Current patterns: Markdown `![alt](path)` and HTML `<img src="...">`
- Use regex in existing pattern list to add new formats
- Ensure moved files end up in correct directory based on match result
- Test with `test_rewrite_markdown_*` test cases

### Debug Image Matching Issues
1. Check logs for tier 1 (MD5) mismatches - usually indicates corrupted extraction
2. Verify PIL/imagehash installed: `python -c "import imagehash"` (returns 9999 if missing)
3. For Ollama tests, ensure ollama service running: `ollama list`
4. Enable debug logging: Set logger level to DEBUG to see detailed comparison info

## Repository Structure Reference
- `pdf2foundry/` - Main package
  - `import_pdf.py` - **Core processing pipeline** (primary work location)
  - `models.py` - Model configuration and tier definitions
  - `processor.py`, `systems.py`, `add_to_module.py` - Planned future implementations (not active)
- `tests/` - Unit tests (run before any changes)
- `docs/` - MkDocs documentation
- `.github/` - GitHub workflows and configuration
- `tox.ini` - Test automation across Python versions
