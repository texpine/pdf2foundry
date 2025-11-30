import glob
import hashlib
import logging
import os
import re
import shutil
import subprocess
from typing import List, Optional

# Mandatory dependencies
import imagehash
import ollama
from PIL import Image


def run_marker_single(
    pdf_path: str,
    output_dir: str,
    llm_service: str,
    llm_model_name: str,
    llm_api_or_url: str,
    logger: logging.Logger,
) -> Optional[str]:
    """Run the `marker_single` CLI to convert a PDF to markdown.

    Returns the path to the generated markdown file on success, or `None` on failure.
    """
    base_name = os.path.basename(pdf_path)
    md_name = os.path.splitext(base_name)[0]
    md_filename = md_name + ".md"
    output_filepath = os.path.join(output_dir, md_name, md_filename)

    marker_cmd = [
        "marker_single",
        pdf_path,
        "--output_format",
        "markdown",
        "--output_dir",
        output_dir,
        "--use_llm",
        "--llm_service",
        llm_service,
        "--ollama_model",
        llm_model_name,
        "--ollama_base_url",
        llm_api_or_url,
        "--paginate_output"
        # "--workers",
        # "4",
    ]

    try:
        logger.info("Running marker_single CLI to generate markdown...")
        res = subprocess.run(marker_cmd, capture_output=True, text=True, check=True)
        logger.info("marker_single finished: %s", (res.stdout or res.stderr)[:200])
    except subprocess.CalledProcessError as e:
        logger.error("marker_single failed: %s", e.stderr or e.stdout)
        print("Error: marker_single CLI failed. See log for details.")
        return None

    if not os.path.exists(output_filepath):
        logger.error("Expected markdown not found at %s", output_filepath)
        print(f"Conversion completed but markdown not found: {output_filepath}")
        return None

    return output_filepath


def run_pdfimages_into_dir(
    pdf_path: str, images_dir: str, logger: logging.Logger
) -> List[str]:
    """Run `pdfimages` with the given images_dir as prefix and return list of created PNG base names.

    If extraction fails or no images were created, returns an empty list.
    """
    pdfimages_cmd = ["pdfimages", "-png", pdf_path, images_dir]
    try:
        logger.info("Extracting images with pdfimages into %s...", images_dir)
        res = subprocess.run(pdfimages_cmd, capture_output=True, text=True, check=True)
        logger.info("pdfimages finished: %s", (res.stdout or res.stderr)[:200])
    except subprocess.CalledProcessError as e:
        logger.error("pdfimages failed: %s", e.stderr or e.stdout)
        print(
            "Warning: pdfimages failed to extract images. Continuing, but images may be missing."
        )
        return []

    # Collect any PNGs created using the provided prefix/directory
    pattern = os.path.join(images_dir, "*")
    created = [
        os.path.basename(p) for p in glob.glob(pattern) if p.lower().endswith(".png")
    ]
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


def ollama_compare_images(img_a: str, img_b: str, logger: logging.Logger) -> bool:
    """Use ollama llava:34b to compare two images.
    Returns True if images are visually similar, False otherwise.
    Requires ollama to be running locally.
    """
    try:
        import base64

        with open(img_a, "rb") as f:
            img_a_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(img_b, "rb") as f:
            img_b_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = "Are these two images showing the same content or very similar content? Answer only 'yes' or 'no'."
        response = ollama.generate(
            model="llava:34b",
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
    abs_candidate: str, png_paths: List[str], logger: logging.Logger
) -> Optional[str]:
    """Find a matching PNG for `abs_candidate` from `png_paths`.

    Matching strategy:
    - Exact MD5 match
    - Perceptual hash (phash) distance (threshold <= 5)
    - Ollama visual comparison as a last resort

    Returns the path to the matched PNG, or None if no match found.
    """
    logger.info("Comparing JPEgs and PNGs" )
    # Compute source MD5 once
    try:
        src_md5 = md5_file(abs_candidate)
    except Exception:
        src_md5 = None

    # Iterate once over png_paths and run checks in order: MD5 -> phash -> Ollama
    for p in png_paths:
        # MD5 exact match
        try:
            if src_md5 is not None and src_md5 == md5_file(p):
                if logger:
                    logger.debug("MD5 match: %s == %s", abs_candidate, p)
                return p
        except Exception:
            pass

        # Perceptual hash (phash) distance
        try:
            d = phash_distance(abs_candidate, p, logger)
            # threshold: accept small distances (tunable)
            if d <= 50:
                if logger:
                    logger.debug("phash match (dist=%s): %s ~= %s", d, abs_candidate, p)
                return p
        except Exception:
            pass

        # Ollama visual comparison as a last resort for this candidate
        try:
            if ollama_compare_images(abs_candidate, p, logger):
                if logger:
                    logger.debug("Ollama visual match: %s ~= %s", abs_candidate, p)
                return p
        except Exception:
            pass

    return None


def rewrite_markdown_image_references(
    md_text: str,
    output_dir: str,
    images_dir: str,
    # png_paths: List[str],
    logger: logging.Logger,
) -> str:
    """Process markdown image references: match with PNGs, move originals, update references.

    Args:
        md_text: The markdown content.
        output_dir: The base output directory.
        images_dir: The directory where PNGs are stored.
        png_paths: List of full paths to PNG files.
        logger: Logger for debug output.

    Returns:
        Updated markdown text with corrected image references.
    """
    # Find image references of the forms: ![alt](path) and <img src="path">
    img_pattern_md = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    img_pattern_html = re.compile(r"<img\s+[^>]*src=[\"']([^\"']+)[\"'][^>]*>")

    def normalize_path(p):
        # Remove leading ./ or / and normalize separators
        return os.path.normpath(p).lstrip("./\\")

    # Get the directory where the markdown file is located (same as images_dir)
    # md_dir = os.path.dirname(images_dir)
    
    # Build list of PNG paths
    # Fallback: if pdfimages didn't create anything, look for any PNGs directly in md_dir
    # if not image_files:
    png_paths = [
        os.path.join(images_dir, os.path.basename(p)) for p in glob.glob(os.path.join(images_dir, "*.png"))
    ]
    
    # Create images_replaced directory in the same directory as the markdown file
    images_replaced_dir = os.path.join(output_dir, "images_replaced/")
    os.makedirs(images_replaced_dir, exist_ok=True)

    # Process all image references
    for match in list(img_pattern_md.finditer(md_text)) + list(
        img_pattern_html.finditer(md_text)
    ):
        img_ref = match.group(1)
        norm = normalize_path(img_ref)
        # Handle absolute paths properly
        if os.path.isabs(norm):
            abs_candidate = norm
        else:
            abs_candidate = os.path.join(output_dir, norm)
        if not os.path.exists(abs_candidate):
            continue

        # Find best matching PNG using helper (MD5 -> phash -> Ollama)
        matched_png = find_matching_png(abs_candidate, png_paths, logger)

        if matched_png:
            # Move original to images_replaced/ with "_replaced" suffix
            # png_basename_no_ext = os.path.splitext(os.path.basename(matched_png))[0]
            orig_basename = os.path.basename(norm)
            # orig_ext = os.path.splitext(orig_basename)[1]

            # renamed_file = f"{png_basename_no_ext}_replaced{orig_ext}"
            dest_orig = os.path.join(images_replaced_dir, orig_basename)
            try:
                shutil.move(abs_candidate, dest_orig)
                logger.info("Moved replaced image %s to %s", abs_candidate, dest_orig)
            except Exception as e:
                logger.warning(
                    "Failed to move original image %s to images_replaced: %s",
                    abs_candidate,
                    e,
                )
                # Don't update markdown reference if move failed
                continue

            # Update markdown reference to use the PNG
            png_basename = os.path.basename(matched_png)
            new_ref = os.path.join("images", png_basename).replace("\\", "/")
            # Replace only the specific match to avoid affecting other references with same filename
            start, end = match.span(1)
            md_text = md_text[:start] + new_ref + md_text[end:]
            logger.info("Replaced image reference %s with %s", img_ref, new_ref)
        else:
            # No match found: keep original behavior (move into images/)
            dest_name = os.path.basename(norm)
            dest_path = os.path.join(images_dir, dest_name)
            try:
                shutil.move(abs_candidate, dest_path)
                new_ref = os.path.join("images", dest_name).replace("\\", "/")
                # Replace only the specific match to avoid affecting other references with same filename
                start, end = match.span(1)
                md_text = md_text[:start] + new_ref + md_text[end:]
                logger.info("Moved unmatched image %s to images/", abs_candidate)
            except Exception as e:
                logger.warning("Could not move image %s: %s", abs_candidate, e)

    return md_text


def convert_pdf_to_markdown(
    pdf_path: str, output_dir: str, llm_service: str, llm_model_name: str, llm_api_or_url: str
) -> None:
    """
    Converts a PDF file to Markdown using Marker.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory to save the output Markdown file.
        llm_service (str): The LLM service class path.
        llm_model_name (str): The model name to use.
        llm_api_or_url (str): The API endpoint or URL.
    """
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
        pdf_path, output_dir, llm_service, llm_model_name, llm_api_or_url, logger
    )

    if not output_filepath:
        # Error already logged in helper
        return

    # Get the directory where the markdown file is located (same as the pdfimages prefix)
    md_dir = os.path.dirname(output_filepath)
    
    # Prepare images directory in the same directory as the markdown file (inside the md file's directory)
    images_dir = os.path.join(md_dir, "images/")
    os.makedirs(images_dir, exist_ok=True)

    # Extract images using helper; returns list of created PNG base names (or empty list)
    image_files = run_pdfimages_into_dir(pdf_path, images_dir, logger)

    # Load markdown
    with open(output_filepath, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Rewrite image references using helper
    md_text = rewrite_markdown_image_references(
        md_text, md_dir, images_dir, logger
    )

    # Save markdown
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(md_text)
        logger.info("Saved updated markdown to %s", output_filepath)
    except Exception as e:
        logger.error("Failed to save updated markdown: %s", e)
        print(f"Error: could not save updated markdown file: {e}")
        return

    print(f"Successfully converted '{pdf_path}' to '{output_filepath}'")


if __name__ == "__main__":
    convert_pdf_to_markdown(
        # "/root/pdf2foundry_input/Skull Wizards of the Chaos Caverns.pdf",
        "/root/pdf2foundry_input/NumeneraDiscovery-Corebook.pdf",
        "/root/pdf2foundry_output/",
        "marker.services.ollama.OllamaService",
        "gpt-oss:120b",
        "http://localhost:11434",
    )

# Export public API
__all__ = [
    "run_marker_single",
    "run_pdfimages_into_dir",
    "md5_file",
    "phash_distance",
    "ollama_compare_images",
    "find_matching_png",
    "rewrite_markdown_image_references",
    "convert_pdf_to_markdown",
]
