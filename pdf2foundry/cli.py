"""Console script for pdf2foundry."""

import fire


def help() -> None:
    print("pdf2foundry")
    print("=" * len("pdf2foundry"))
    print(
        "Tool that converts a TTRPG book in the PDF format into a module for Foundry with the content turned into Compendiums and more."
    )


def main() -> None:
    fire.Fire({"help": help})


if __name__ == "__main__":
    main()  # pragma: no cover
