import os
import shutil
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageDraw

import pdf2foundry.import_pdf as impdf


class TestFindMatchingPNG(unittest.TestCase):
    def test_md5_match(self, logger):
        with tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "img1.png")
            candidate = os.path.join(td, "orig.png")
            im = Image.new("RGB", (32, 32), "white")
            im.save(png)
            shutil.copyfile(png, candidate)

            res = impdf.find_matching_png(candidate, [png], logger=logger)
            self.assertEqual(res, png)

    def test_phash_match(self, logger):
        with tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "img1.png")
            candidate = os.path.join(td, "orig.png")

            im1 = Image.new("RGB", (64, 64), "white")
            draw1 = ImageDraw.Draw(im1)
            draw1.rectangle([10, 10, 20, 20], fill="black")
            im1.save(png)

            im2 = Image.new("RGB", (64, 64), "white")
            draw2 = ImageDraw.Draw(im2)
            draw2.rectangle([11, 10, 21, 20], fill="black")
            im2.save(candidate)

            res = impdf.find_matching_png(candidate, [png], logger=logger)
            self.assertEqual(res, png)

    def test_ollama_fallback(self, logger):
        with tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "img1.png")
            candidate = os.path.join(td, "orig.png")
            im = Image.new("RGB", (32, 32), "white")
            im.save(png)
            im2 = Image.new("RGB", (32, 32), "black")
            im2.save(candidate)

            # Patch ollama_compare_images to simulate a positive visual match
            with mock.patch.object(
                impdf, "ollama_compare_images", return_value=True
            ) as patched:
                res = impdf.find_matching_png(candidate, [png], logger=logger)
                self.assertEqual(res, png)
                patched.assert_called()


if __name__ == "__main__":
    unittest.main()
