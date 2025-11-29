# PDF2Foundry Tests

Comprehensive unit test suite for the image processing and markdown rewriting functionality.

## Quick Start

### Prerequisites
Install test dependencies:
```bash
pip install -r ../requirements.txt  # or follow setup/install_python.sh
```

### Run All Tests
```bash
python -m unittest discover -s . -p "test_*.py" -v
```

### Run Specific Test File
```bash
python -m unittest test_import_pdf -v
```

### Run Specific Test Class
```bash
python -m unittest test_import_pdf.TestRewriteMarkdownImageReferences -v
```

### Run Specific Test Method
```bash
python -m unittest test_import_pdf.TestMD5File.test_md5_file_consistency -v
```

## Test Coverage

| Module | Class | Test Cases | Coverage |
|--------|-------|-----------|----------|
| `import_pdf.py` | `TestMD5File` | 3 | Hashing, consistency, error handling |
| | `TestPhashDistance` | 2 | Perceptual hash, missing dependencies |
| | `TestOllamaCompareImages` | 3 | Ollama integration, model response parsing |
| | `TestRewriteMarkdownImageReferences` | 8 | Markdown rewriting, file moves, image matching |
| **Total** | | **16** | Core functionality and edge cases |

## Test Categories

### Unit Tests (Isolated Functions)
- `md5_file()` – File hashing
- `phash_distance()` – Perceptual hash computation
- `ollama_compare_images()` – LLM-based comparison

### Integration Tests (Markdown Rewriting)
- Markdown image reference detection (both `![alt](url)` and `<img src="">` formats)
- Image matching logic (MD5, phash, ollama fallback chain)
- File organization (moving to `images/` vs `images_replaced/`)
- Renaming with `_replaced` suffix
- Path normalization

### Edge Cases
- Missing dependencies (graceful fallback)
- Non-existent files
- Relative paths (`./ `prefix)
- Multiple image references
- Unmatched images

## Common Issues

### ImportError: No module named 'pdf2foundry'
Ensure you're in the repo root and running:
```bash
python -m unittest tests.test_import_pdf -v
```
Or add the parent directory to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m unittest tests.test_import_pdf -v
```

### Missing PIL/imagehash warnings
These are optional dependencies. Tests should still pass with fallback behavior.

### Ollama not available
Ollama tests mock the library, so tests pass without a running Ollama instance.

## Extending Tests

Add new test methods to existing classes or create new test classes:

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup test environment
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Cleanup
        shutil.rmtree(self.temp_dir)

    def test_new_behavior(self):
        # Your test here
        self.assertTrue(expected_result)

if __name__ == "__main__":
    unittest.main()
```

Run discovery to automatically pick up new tests:
```bash
python -m unittest discover -s . -p "test_*.py" -v
```
