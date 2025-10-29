import logging
from pathlib import Path

import arxiv
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Download COVID-19 papers from arXiv and extract text."""
    base_dir = Path(__file__).parent.parent / "data" / "covid_papers"
    pdf_dir = base_dir / "pdf"
    txt_dir = base_dir / "txt"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    search = arxiv.Search(
        query="covid-19 OR coronavirus OR SARS-CoV-2",
        max_results=100,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    client = arxiv.Client()

    for result in client.results(search):
        paper_id = result.entry_id.split("/")[-1].replace(".", "_")
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        txt_path = txt_dir / f"{paper_id}.txt"

        if txt_path.exists():
            logger.info(f"Skipping {paper_id} (already processed)")
            continue

        logger.info(f"Downloading {paper_id}...")
        result.download_pdf(dirpath=str(pdf_dir), filename=f"{paper_id}.pdf")

        logger.info(f"Extracting text from {paper_id}...")
        try:
            with pdf_path.open("rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

                with txt_path.open("w", encoding="utf-8") as txt_file:
                    txt_file.write(text)

                logger.info(f"Successfully processed {paper_id}")
        except (OSError):
            logger.exception(f"Error processing {paper_id}")

    logger.info("Download and extraction complete!")


if __name__ == "__main__":
    main()
