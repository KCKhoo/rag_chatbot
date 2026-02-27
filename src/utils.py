from pathlib import Path


def load_faqs(faq_folder_path: str) -> list[str]:
    """
    Load FAQ documents in Markdown format from a specified folder

    Assumptions:
        - All files in the provided folder are Markdown files.
        - Each file represents a single FAQ document.

    :param faq_folder_path: Path to the folder containing all the FAQs
    :return: List of FAQ contents
    """

    folder = Path(faq_folder_path)

    contents = []
    for file_path in folder.iterdir():
        with file_path.open("r", encoding="utf-8") as f:
            contents.append(f.read())

    return contents
