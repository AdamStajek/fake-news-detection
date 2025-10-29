import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


def fetch_text_from_url(url: str, timeout: int = 10) -> str | None:
    """Fetch and extract plain text from a URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)
    except requests.RequestException:
        return None


def generate_filename(url: str, index: int) -> str:
    """Generate a filename from URL and index."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "").replace(".", "_")
    return f"{index:04d}_{domain}.txt"


def main() -> None:
    """Fetch text from first 200 MMCovid dataset URLs and save as files."""
    project_root = Path(__file__).parent.parent.parent
    csv_path = (
        project_root / "evaluation" / "data" / "mmcovid" / "english_news.csv"
    )
    output_dir = (
        project_root
        / "knowledge_base"
        / "data"
        / "specific_covid_facts_from_mmcovid"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    df_with_urls = df[df["fact_url"].notna()].head(200)

    for idx, row in df_with_urls.iterrows():
        url = row["fact_url"]
        index_num = int(idx) if isinstance(idx, (int, float)) else 0
        filename = generate_filename(url, index_num)
        filepath = output_dir / filename

        if filepath.exists():
            continue

        text = fetch_text_from_url(url)

        if text:
            filepath.write_text(
                f"URL: {url}\n"
                f"Claim: {row.get('claim', 'N/A')}\n"
                f"Label: {row.get('label', 'N/A')}\n"
                f"{'=' * 80}\n\n"
                f"{text}",
                encoding="utf-8",
            )

        time.sleep(1)


if __name__ == "__main__":
    main()




