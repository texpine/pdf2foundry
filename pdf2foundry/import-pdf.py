import os
import sys

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


def convert_pdf_to_markdown(pdf_path, output_dir="output_markdown"):
    """
    Converts a PDF file to Markdown using Marker.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory to save the output Markdown file.
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

    print(f"Starting conversion for: {pdf_path}")
    # try:
    # Initialize the converter
    # It might take time to load models on the first run
    converter = PdfConverter(artifact_dict=create_model_dict())

    # Convert the PDF
    # The converter returns a complex object, not just text directly
    rendered_output = converter(pdf_path)

    # Construct the output path
    base_name = os.path.basename(pdf_path)
    md_filename = os.path.splitext(base_name)[0] + ".md"
    output_filepath = os.path.join(output_dir, md_filename)

    # Save the markdown output
    markdown_text = rendered_output.markdown
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    # Save the images
    images = rendered_output.images  # Access the images attribute
    if images:
        print(f"Found {len(images)} images to save.")
        for img_filename, img_bytes in images.items():
            img_obj = img_bytes  # Rename for clarity, it's an Image object
            img_output_path = os.path.join(output_dir, img_filename)
            try:
                # Use the Image object's save method
                img_obj.save(img_output_path)
                # print(f"Saved image: {img_output_path}") # Optional: uncomment for verbose output
            except Exception as img_e:
                print(f"Error saving image {img_filename}: {img_e}")

    if os.path.exists(output_filepath):
        print(f"Successfully converted '{pdf_path}' to '{output_filepath}'")
    else:
        print(
            f"Conversion process completed, but the expected output file was not found: {output_filepath}"
        )
        print("Please check the console for potential errors from Marker.")

    # except Exception as e:
    #     print(f"An error occurred during conversion: {e}")


if __name__ == "__main__":
    convert_pdf_to_markdown(
        "../pdf2foundry_input/Skull Wizards of the Chaos Caverns.pdf",
        "../pdf2foundry_output/",
    )
