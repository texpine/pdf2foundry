"""Unit tests for pdf2foundry.import_pdf module."""

import hashlib
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path to import pdf2foundry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pdf2foundry.import_pdf import (
    md5_file,
    ollama_compare_images,
    phash_distance,
    rewrite_markdown_image_references,
)


class TestMD5File(unittest.TestCase):
    """Test md5_file function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_md5_file_consistency(self):
        """Test that md5_file returns consistent hash for same file."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        hash1 = md5_file(test_file)
        hash2 = md5_file(test_file)
        self.assertEqual(hash1, hash2)

    def test_md5_file_different_content(self):
        """Test that md5_file returns different hash for different content."""
        file1 = os.path.join(self.temp_dir, "file1.txt")
        file2 = os.path.join(self.temp_dir, "file2.txt")

        with open(file1, "w") as f:
            f.write("content1")
        with open(file2, "w") as f:
            f.write("content2")

        hash1 = md5_file(file1)
        hash2 = md5_file(file2)
        self.assertNotEqual(hash1, hash2)

    def test_md5_file_nonexistent(self):
        """Test that md5_file raises error for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            md5_file(os.path.join(self.temp_dir, "nonexistent.txt"))


class TestPhashDistance(unittest.TestCase):
    """Test phash_distance function."""

    def setUp(self):
        self.logger = logging.getLogger("test")

    def test_phash_distance_error_handling(self):
        """Test that phash_distance returns 9999 on error."""
        # Pass invalid file path
        distance = phash_distance(
            "invalid_path.png", "another_invalid.png", self.logger
        )
        self.assertEqual(distance, 9999)


class TestOllamaCompareImages(unittest.TestCase):
    """Test ollama_compare_images function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger("test")
        # Create dummy image files
        self.img1 = os.path.join(self.temp_dir, "img1.png")
        self.img2 = os.path.join(self.temp_dir, "img2.png")
        with open(self.img1, "wb") as f:
            f.write(b"fake image 1")
        with open(self.img2, "wb") as f:
            f.write(b"fake image 2")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_ollama_compare_images_returns_bool(self):
        """Test that ollama_compare_images returns a boolean."""
        with patch("pdf2foundry.import_pdf.ollama") as mock_ollama:
            mock_ollama.generate.return_value = {"response": "yes"}
            result = ollama_compare_images(self.img1, self.img2, self.logger)
            self.assertTrue(result)

    def test_ollama_compare_images_false_on_no(self):
        """Test that ollama_compare_images returns False when model says no."""
        with patch("pdf2foundry.import_pdf.ollama") as mock_ollama:
            mock_ollama.generate.return_value = {"response": "no"}
            result = ollama_compare_images(self.img1, self.img2, self.logger)
            self.assertFalse(result)


class TestRewriteMarkdownImageReferences(unittest.TestCase):
    """Test rewrite_markdown_image_references function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger("test")

        # Create directories
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir)

        # Create test image files
        self.img1_path = os.path.join(self.output_dir, "original_image.jpg")
        self.png1_path = os.path.join(self.images_dir, "extracted001.png")

        with open(self.img1_path, "wb") as f:
            f.write(b"original image content")
        with open(self.png1_path, "wb") as f:
            f.write(b"original image content")  # Same as original for MD5 match

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_rewrite_markdown_md_image_reference(self):
        """Test that markdown image references are found and rewritten."""
        md_text = "# Document\n![alt](original_image.jpg)\n"
        png_paths = [self.png1_path]

        result = rewrite_markdown_image_references(
            md_text, self.output_dir, self.images_dir, png_paths, self.logger
        )

        # Should contain reference to PNG
        self.assertIn("extracted001.png", result)
        self.assertNotIn("original_image.jpg", result)

    def test_rewrite_markdown_html_image_reference(self):
        """Test that HTML image references are found and rewritten."""
        md_text = '<img src="original_image.jpg" />\n'
        png_paths = [self.png1_path]

        result = rewrite_markdown_image_references(
            md_text, self.output_dir, self.images_dir, png_paths, self.logger
        )

        # Should contain reference to PNG
        self.assertIn("extracted001.png", result)
        self.assertNotIn("original_image.jpg", result)

    def test_rewrite_markdown_images_replaced_folder_created(self):
        """Test that images_replaced folder is created."""
        md_text = "![alt](original_image.jpg)\n"
        png_paths = [self.png1_path]

        rewrite_markdown_image_references(
            md_text, self.output_dir, self.images_dir, png_paths, self.logger
        )

        images_replaced_dir = os.path.join(self.output_dir, "images_replaced")
        self.assertTrue(os.path.exists(images_replaced_dir))

    def test_rewrite_markdown_original_file_moved_with_suffix(self):
        """Test that original file is moved to images_replaced with _replaced suffix."""
        md_text = "![alt](original_image.jpg)\n"
        png_paths = [self.png1_path]

        # Original file should exist before
        self.assertTrue(os.path.exists(self.img1_path))

        rewrite_markdown_image_references(
            md_text, self.output_dir, self.images_dir, png_paths, self.logger
        )

        # Original file should be moved
        self.assertFalse(os.path.exists(self.img1_path))

        # Replaced file should exist with _replaced suffix
        replaced_path = os.path.join(
            self.output_dir, "images_replaced", "extracted001_replaced.jpg"
        )
        self.assertTrue(os.path.exists(replaced_path))

    def test_rewrite_markdown_no_match_moves_to_images(self):
        """Test that unmatched images are moved to images/ folder."""
        md_text = "![alt](original_image.jpg)\n"
        png_paths = []  # No PNGs to match

        # Create a fresh image file for this test
        img_path = os.path.join(self.output_dir, "unmatched.jpg")
        with open(img_path, "wb") as f:
            f.write(b"unmatched image")

        md_text_with_unmatched = "![alt](unmatched.jpg)\n"

        result = rewrite_markdown_image_references(
            md_text_with_unmatched,
            self.output_dir,
            self.images_dir,
            png_paths,
            self.logger,
        )

        # Image should be moved to images/
        moved_path = os.path.join(self.images_dir, "unmatched.jpg")
        self.assertTrue(os.path.exists(moved_path))
        self.assertFalse(os.path.exists(img_path))

        # Markdown should reference the new location
        self.assertIn("images/unmatched.jpg", result)

    def test_rewrite_markdown_normalizes_paths(self):
        """Test that paths are normalized correctly."""
        # Create image with ./ prefix
        md_text = "![alt](./original_image.jpg)\n"
        png_paths = [self.png1_path]

        result = rewrite_markdown_image_references(
            md_text, self.output_dir, self.images_dir, png_paths, self.logger
        )

        # Original file should still be moved
        self.assertFalse(os.path.exists(self.img1_path))
        # Should reference PNG
        self.assertIn("extracted001.png", result)

    def test_rewrite_markdown_multiple_references(self):
        """Test handling multiple image references in markdown."""
        # Create two PNG files
        png2_path = os.path.join(self.images_dir, "extracted002.png")
        img2_path = os.path.join(self.output_dir, "image2.jpg")

        with open(png2_path, "wb") as f:
            f.write(b"image2 content")
        with open(img2_path, "wb") as f:
            f.write(b"image2 content")

        md_text = "![img1](original_image.jpg)\nSome text\n![img2](image2.jpg)\n"
        png_paths = [self.png1_path, png2_path]

        result = rewrite_markdown_image_references(
            md_text, self.output_dir, self.images_dir, png_paths, self.logger
        )

        # Both should be replaced
        self.assertIn("extracted001.png", result)
        self.assertIn("extracted002.png", result)

        # Both original files should be moved
        self.assertFalse(os.path.exists(self.img1_path))
        self.assertFalse(os.path.exists(img2_path))


if __name__ == "__main__":
    unittest.main()
