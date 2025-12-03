import pdfplumber
import pandas as pd
from typing import List, Tuple


def extract_tables_from_pdf(filepath: str) -> List[Tuple[int, pd.DataFrame]]:
    """
    Extract tables from a PDF using pdfplumber.
    
    Returns:
        List of (page_number, dataframe)
    """
    tables = []

    with pdfplumber.open(filepath) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                extracted_tables = page.extract_tables()

                for t in extracted_tables:
                    # Convert table (list of lists) to pandas DataFrame
                    df = pd.DataFrame(t)

                    # Remove completely empty columns
                    df = df.dropna(axis=1, how="all")

                    # Clean header row if necessary
                    df = df.rename(columns=df.iloc[0]).drop(df.index[0])

                    tables.append((page_number, df))

            except Exception:
                continue

    return tables


def table_to_tsv_string(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a TSV text block.
    This text block can be embedded and used in retrieval.
    """
    return df.to_csv(sep="\t", index=False)
