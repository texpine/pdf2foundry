import base64
import binascii
import hashlib
import io
import logging
import os
import platform
import re
import shutil
import sys
from typing import List, Optional
import fitz
import imagehash
import ollama
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL import Image
import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim

def is_debugging() -> bool:
    """Check if the script is running under a debugger.

    Checks multiple indicators:
    - sys.gettrace() for trace-based debuggers
    - PYTHONBREAKPOINT environment variable
    - Common debugger modules in sys.modules
    """
    # Check if trace function is set (pdb, debugpy, etc.)
    if sys.gettrace() is not None:
        return True

    # Check for debugger environment variables
    if os.environ.get("PYTHONBREAKPOINT", "") not in ("", "0"):
        return True

    # Check if running under common debuggers
    debugger_modules = {"pydevd", "debugpy", "pdb"}
    if debugger_modules & set(sys.modules.keys()):
        return True

    return False


def run_marker_single(
    pdf_path: str,
    output_dir: str,
    logger: logging.Logger,
    llm_service: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    llm_api_or_url: Optional[str] = None,
    vision_service: Optional[str] = None,
    vision_model_name: Optional[str] = None,
    vision_api_or_url: Optional[str] = None,
) -> Optional[str]:
    """Run the `marker_single` CLI to convert a PDF to markdown.

    Returns the path to the generated markdown file on success, or `None` on failure.
    """
    base_name = os.path.basename(pdf_path)
    md_name = os.path.splitext(base_name)[0]
    md_filename = md_name + ".md"
    output_filepath = os.path.join(output_dir, md_name, md_filename)

    config = {
        "output_format": "markdown",
        "paginate_output": True,
    }

    if llm_service is not None:
        config["use_llm"] = True
        config["llm_service"] = llm_service
        config["ollama_model"] = llm_model_name
        config["ollama_base_url"] = llm_api_or_url
    else:
        config["use_llm"] = False

    try:
        logger.info("Running Marker conversion to generate markdown...")
        config_parser = ConfigParser(config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(pdf_path)
        md_text, _, images = text_from_rendered(rendered)
        logger.info("Marker conversion finished")
    except Exception as e:
        logger.error("Marker conversion failed: %s", e)
        print("Error: Marker conversion failed. See log for details.")
        return None

    # Create output directory
    md_dir = os.path.dirname(output_filepath)
    os.makedirs(md_dir, exist_ok=True)

    def path_relative_to_md(abs_path: str):
        return os.path.relpath(abs_path, start=md_dir).replace(os.path.sep, "/")

    # Save markdown
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(md_text)
    # Prepare images directory
    images_dir = os.path.join(md_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    # Extract images using run_pdfimages_into_dir
    better_png_paths = run_pdfimages_into_dir(pdf_path, images_dir, logger)
    # Create images_replaced directory
    images_replaced_dir = os.path.join(md_dir, "images_replaced/")
    os.makedirs(images_replaced_dir, exist_ok=True)
    # Create a mapping of original RELATIVE image paths to better RELATIVE PNG paths, or to RELATIVE storage dir
    relative_paths_mapping = {}
    # Save images (accept multiple possible formats returned by Marker)
    for rel_path, img_obj in images.items():
        img_path = os.path.join(md_dir, rel_path)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        try:
            # base64 encoded string
            if isinstance(img_obj, str):
                try:
                    data = base64.b64decode(img_obj)
                    with open(img_path, "wb") as f:
                        f.write(data)
                except (binascii.Error, TypeError):
                    logger.warning(
                        "Image for %s is a string but not base64 encoded; skipping",
                        rel_path,
                    )

            # raw bytes-like
            elif isinstance(img_obj, (bytes, bytearray)):
                with open(img_path, "wb") as f:
                    f.write(img_obj)

            # memoryview
            elif isinstance(img_obj, memoryview):
                with open(img_path, "wb") as f:
                    f.write(img_obj.tobytes())

            # PIL Image object
            elif isinstance(img_obj, Image.Image):
                # infer format from extension if possible
                ext = os.path.splitext(rel_path)[1].lstrip(".").upper() or "PNG"
                try:
                    img_obj.save(img_path, format=ext)
                except Exception:
                    img_obj.save(img_path)

            else:
                # fallback: try to base64-decode whatever it is
                try:
                    data = base64.b64decode(img_obj)
                    with open(img_path, "wb") as f:
                        f.write(data)
                except Exception:
                    logger.warning(
                        "Unrecognized image data type for %s; skipping", rel_path
                    )

            # Find best matching PNG using helper (MD5 -> phash -> Ollama)
            matched_png = find_matching_png(
                img_path,
                better_png_paths,
                logger,
                vision_service,
                vision_model_name,
                vision_api_or_url,
            )
            img_path_basename = os.path.basename(img_path)
            if matched_png:
                matched_png_basename = os.path.basename(matched_png)
                matched_png_basename_no_ext = os.path.splitext(matched_png_basename)[0]
                # Create directory for replaced images
                move_to_dir = os.path.join(
                    images_replaced_dir, f"{matched_png_basename_no_ext}"
                )
                os.makedirs(move_to_dir, exist_ok=True)
                # Move original image to images_replaced/<better png file name>/
                img_path_after_replaced = os.path.join(move_to_dir, img_path_basename)
                try:
                    shutil.move(img_path, img_path_after_replaced)
                    # Add to mapping
                    # get relative path to the markdown file's folder
                    # normalize path to forward slashes (good for Markdown and cross-platform)
                    relative_paths_mapping[rel_path] = path_relative_to_md(matched_png)
                    logger.info(
                        "Moved replaced image %s to %s", img_path_basename, move_to_dir
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to move original image %s to images_replaced: %s",
                        img_path,
                        e,
                    )
            else:
                # No match found: keep original behavior and move into images/ because the original files are created in the same folder of the .md
                # dest_name = os.path.basename(img_path)
                dest_path = os.path.join(images_dir, img_path_basename)
                try:
                    shutil.move(img_path, dest_path)
                    # Add to mapping
                    relative_paths_mapping[rel_path] = path_relative_to_md(dest_path)
                    logger.info(
                        "Moved unmatched image %s to images/", img_path_basename
                    )
                except Exception as e:
                    logger.warning("Could not move image %s: %s", img_path_basename, e)

        except Exception as e:
            logger.warning("Failed to save image %s: %s", rel_path, e)

    # Update all markdown references in a single pass
    for old_rel_path, new_rel_path in relative_paths_mapping.items():
        # img_pattern_md = re.compile(r"!\[[^\]]*\]\(([^)]+)\)"
        img_pattern_md = re.compile(
            rf"!\[(?=[^\]]*{old_rel_path}[^\]]*\]|[^\]]*\]\([^)]*{old_rel_path})([^\]]*)\]\(([^)]+)\)"
        )
        # img_pattern_html = re.compile(r"<img\s+[^>]*src=[\"']([^\"']+)[\"'][^>]*>")
        img_pattern_html = re.compile(
            rf"<img\s+(?=[^>]*{old_rel_path})[^>]*src=[\"']([^\"']+)[\"'][^>]*>"
        )
        # Collect all references to this image in the markdown
        image_refs = list(img_pattern_md.finditer(md_text)) + list(
            img_pattern_html.finditer(md_text)
        )
        # final replacement of image reference in the MD code
        for match in image_refs:
            img_ref = match.group(0)
            md_text = md_text.replace(
                img_ref, f"![[{new_rel_path}]]"
            )  # markdown links that work in Obsidian
            logger.info("Replaced image reference %s with %s", img_ref, new_rel_path)

    # Save updated markdown
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(md_text)

    return output_filepath


def run_pdfimages_into_dir(
    pdf_path: str, images_dir: str, logger: logging.Logger
) -> List[str]:
    """Run `pdfimages` with the given images_dir as prefix and return list of created PNG base names.

    If extraction fails or no images were created, returns an empty list.
    """
    # Prefer using PyMuPDF (fitz) to extract embedded images (preserves PNG alpha).
    # Try PyMuPDF (fast, preserves embedded PNGs and alpha channels)
    created = []
    try:
        logger.info("Attempting to extract images via PyMuPDF...")
        doc = fitz.open(pdf_path)
        for pno in range(len(doc)):
            # get_page_images returns tuples; first element is xref
            try:
                images = doc.get_page_images(pno, full=True)
            except Exception:
                images = doc.get_page_images(pno)

            for img in images:
                xref = img[0]
                try:
                    image_dict = doc.extract_image(xref)
                    img_bytes = image_dict.get("image")
                    ext = image_dict.get("ext", "png")
                    out_basename = f"p{pno}_img{xref}.{ext}"
                    out_path = os.path.join(images_dir, out_basename)
                    os.makedirs(images_dir, exist_ok=True)
                    if img_bytes is None:
                        logger.debug("No image bytes for xref %s on page %s", xref, pno)
                        continue
                    if ext.lower() == "png":
                        with open(out_path, "wb") as fh:
                            fh.write(img_bytes)
                    else:
                        # convert other formats to PNG to keep consistent output
                        try:
                            im = Image.open(io.BytesIO(img_bytes))
                            png_basename = os.path.splitext(out_basename)[0] + ".png"
                            out_png = os.path.join(images_dir, png_basename)
                            im.save(out_png, "PNG")
                            out_basename = png_basename
                            out_path = out_png
                        except Exception:
                            # as a last resort write raw bytes with original ext
                            with open(out_path, "wb") as fh:
                                fh.write(img_bytes)

                    created.append(out_path)
                except Exception as e:
                    logger.debug("PyMuPDF failed for page %s xref %s: %s", pno, xref, e)

        doc.close()
        if created:
            logger.info("PyMuPDF extracted %d images", len(created))
        else:
            logger.debug("PyMuPDF found no images to extract")
    except Exception as e:
        logger.debug("PyMuPDF extraction failed or not available: %s", e)

    return created


def md5_file(path: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_distance(a: str, b: str, logger: logging.Logger) -> int:
    """Compute perceptual hash distance between two images."""
    try:
        ha = imagehash.phash(Image.open(a))
        hb = imagehash.phash(Image.open(b))
        return ha - hb
    except Exception as e:
        if logger:
            logger.debug("Perceptual hash comparison failed for %s vs %s: %s", a, b, e)
        return 9999


def hist_bhattacharyya(path_a: str, path_b: str, logger: Optional[logging.Logger] = None) -> float:
    """Compute a Bhattacharyya-like distance between color histograms of two images.

    Returns a float in [0,1] where smaller means more similar. On error returns 1.0.
    """
    try:
        a = Image.open(path_a).convert("RGBA")
        b = Image.open(path_b).convert("RGBA")
        ha = a.histogram()
        hb = b.histogram()
        try:
            import numpy as np
        except Exception:
            np = None
        if np is None:
            s = sum((abs(x - y) for x, y in zip(ha, hb)))
            return float(s) / (sum(ha) + 1)
        ha = np.array(ha, dtype=np.float64)
        hb = np.array(hb, dtype=np.float64)
        ha /= ha.sum() + 1e-12
        hb /= hb.sum() + 1e-12
        bc = np.sum(np.sqrt(ha * hb))
        return 1.0 - float(bc)
    except Exception as e:
        if logger:
            logger.debug("hist_bhattacharyya failed for %s vs %s: %s", path_a, path_b, e)
        return 1.0


def multi_hashes_close(a_path: str, b_path: str, logger: Optional[logging.Logger] = None) -> bool:
    """Compare multiple perceptual hashes (phash, dhash, average_hash).

    Returns True if hashes indicate a close match under stricter thresholds.
    """
    try:
        ph = imagehash.phash(Image.open(a_path)) - imagehash.phash(Image.open(b_path))
        dh = imagehash.dhash(Image.open(a_path)) - imagehash.dhash(Image.open(b_path))
        ah = imagehash.average_hash(Image.open(a_path)) - imagehash.average_hash(Image.open(b_path))
        if ph <= 4 and (dh <= 4 or ah <= 4):
            if logger:
                logger.debug("multi-hash pass ph=%s dh=%s ah=%s for %s vs %s", ph, dh, ah, a_path, b_path)
            return True
    except Exception as e:
        if logger:
            logger.debug("multi_hashes_close failed: %s", e)
    return False


def ssim_similar(a_path: str, b_path: str, logger: Optional[logging.Logger] = None) -> Optional[float]:
    """Compute SSIM between two grayscale images if possible.

    Returns SSIM score in [0,1] or None if unavailable or images differ in size.
    """        
    try:
        a = Image.open(a_path).convert("L")
        b = Image.open(b_path).convert("L")
        if a.size != b.size:
            return None
        aa = np.array(a, dtype=np.uint8)
        bb = np.array(b, dtype=np.uint8)
        score = float(ssim(aa, bb))
        return score
    except Exception as e:
        if logger:
            logger.debug("SSIM compare failed: %s", e)
        return None


def orb_feature_match(a_path: str, b_path: str, logger: Optional[logging.Logger] = None) -> int:
    """Return number of good ORB matches between two images (requires OpenCV)."""
    try:
        img1 = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return len(good)
    except Exception as e:
        if logger:
            logger.debug("ORB matching failed: %s", e)
        return 0


def vision_compare_images(
    img_a: str,
    img_b: str,
    logger: logging.Logger,
    vision_service: str,
    vision_model_name: str,
    vision_api_or_url: str,
) -> bool:
    """Use ollama llava:34b to compare two images.
    Returns True if images are visually similar, False otherwise.
    Requires ollama to be running locally.
    """
    if vision_service != "Ollama":  # only ollama supported so far
        return False

    try:
        import base64

        with open(img_a, "rb") as f:
            img_a_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(img_b, "rb") as f:
            img_b_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = "Are these two images showing the same content or very similar content? Answer only 'yes' or 'no'."
        client = ollama.Client(
            host=vision_api_or_url,  # Change this if using a different host
        )
        response = client.generate(
            model=vision_model_name,
            prompt=prompt,
            images=[img_a_b64, img_b_b64],
            stream=False,
        )
        answer = response.get("response", "").strip().lower()
        return "yes" in answer
    except Exception as e:
        if logger:
            logger.warning("Ollama comparison failed: %s", e)
        return False


def find_matching_png(
    abs_candidate: str,
    png_paths: List[str],
    logger: logging.Logger,
    vision_service: Optional[str] = None,
    vision_model_name: Optional[str] = None,
    vision_api_or_url: Optional[str] = None,
) -> Optional[str]:
    """Find a matching PNG for `abs_candidate` from `png_paths`.

    Matching strategy:
    - Exact MD5 match
    - Perceptual hash (phash) distance (threshold <= 5)
    - Ollama visual comparison as a last resort

    Returns the path to the matched PNG, or None if no match found.
    """
    # Compute source MD5 once
    try:
        src_md5 = md5_file(abs_candidate)
    except Exception:
        src_md5 = None

    # Iterate candidates: cheap -> expensive -> Ollama fallback
    for p in png_paths:
        # 1) MD5 exact match (very fast)
        try:
            if src_md5 is not None and src_md5 == md5_file(p):
                if logger:
                    logger.debug("MD5 match: %s == %s", abs_candidate, p)
                return p
        except Exception:
            pass

        # 2) Quick file-size check (fast) - reject if sizes wildly different
        try:
            a_size = os.path.getsize(abs_candidate)
            b_size = os.path.getsize(p)
            if a_size == b_size:
                # same file size is a good hint; continue to hash checks
                if logger:
                    logger.debug("Filesize equal hint for %s and %s", abs_candidate, p)
        except Exception:
            a_size = b_size = None

        # 3) Perceptual/hash checks (fast)
        try:
            if multi_hashes_close(abs_candidate, p):
                if logger:
                    logger.debug("Perceptual multi-hash match: %s ~= %s", abs_candidate, p)
                return p
        except Exception:
            pass

        # 4) Color histogram/Bhattacharyya (medium)
        try:
            hdist = hist_bhattacharyya(abs_candidate, p)
            # stricter threshold: accept only very close histograms
            if hdist <= 0.12:
                if logger:
                    logger.debug("Histogram BH dist=%.3f for %s vs %s", hdist, abs_candidate, p)
                return p
        except Exception:
            pass

        # 5) SSIM (more expensive) - requires same dimensions
        try:
            s = ssim_similar(abs_candidate, p)
            if s is not None and s >= 0.95:
                if logger:
                    logger.debug("SSIM=%.3f match for %s vs %s", s, abs_candidate, p)
                return p
        except Exception:
            pass

        # 6) ORB keypoint matching (expensive)
        try:
            good = orb_feature_match(abs_candidate, p)
            # stricter acceptance: require at least 12 good matches
            if good >= 12:
                if logger:
                    logger.debug("ORB good matches=%d for %s vs %s", good, abs_candidate, p)
                return p
        except Exception:
            pass

        # 7) Ollama visual comparison as last resort
        try:
            if (
                vision_service
                and vision_model_name
                and vision_api_or_url
                and vision_compare_images(
                    abs_candidate,
                    p,
                    logger,
                    vision_service,
                    vision_model_name,
                    vision_api_or_url,
                )
            ):
                if logger:
                    logger.debug("Ollama visual match: %s ~= %s", abs_candidate, p)
                return p
        except Exception:
            pass
        
    return None


def convert_pdf_to_markdown(
    pdf_path,
    output_dir,
    llm_service: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    llm_api_or_url: Optional[str] = None,
    vision_service: Optional[str] = None,
    vision_model_name: Optional[str] = None,
    vision_api_or_url: Optional[str] = None,
):
    """
    Converts a PDF file to Markdown using Marker.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory to save the output Markdown file.
        llm_service (Optional[str]): The LLM service class path. If None, no LLM is used.
        llm_model_name (Optional[str]): The model name to use. Ignored if llm_service is None.
        llm_api_or_url (Optional[str]): The API endpoint or URL. Ignored if llm_service is None.
    """
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # pdf_path = os.path.abspath(os.path.join(script_dir, pdf_path))
    # output_dir = os.path.abspath(os.path.join(script_dir, output_dir))
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print(f"Starting conversion for: {pdf_path}")

    # Run marker_single via helper function; returns the markdown filepath or None
    output_filepath = run_marker_single(
        pdf_path,
        output_dir,
        logger,
        llm_service,
        llm_model_name,
        llm_api_or_url,
        vision_service,
        vision_model_name,
        vision_api_or_url,
    )

    print(f"Successfully converted '{pdf_path}' to '{output_filepath}'")


if __name__ == "__main__":
    # only use AI in debug mode when on Linux
    if is_debugging() and platform.system() == "Linux":
        llm_service = "Ollama"
        llm_model_name = "gpt-oss:120b"
        llm_api_or_url = "http://localhost:11434"
        vision_service = "Ollama"
        vision_model_name = "llava:34b"
        vision_api_or_url = "http://localhost:11434"
    else:
        llm_service = None
        llm_model_name = None
        llm_api_or_url = None
        vision_service = None
        vision_model_name = None
        vision_api_or_url = None

    convert_pdf_to_markdown(
        # "/Claude/pdf2foundry_input/Skull Wizards of the Chaos Caverns.pdf",
        # "/Claude/pdf2foundry_input/Swords & Wizardry - Black Box Books - Tome 1 - Astronauts and Ancients.pdf",
        "/Claude/pdf2foundry_input/5f-discover-lands-unknown-parts-replacement.pdf",
        # "/Claude/pdf2foundry_input/StarsWithoutNumber-SkywardSteel.pdf",
        "/Claude/pdf2foundry_output/",
        llm_service,
        llm_model_name,
        llm_api_or_url,
        vision_service,
        vision_model_name,
        vision_api_or_url,
    )
