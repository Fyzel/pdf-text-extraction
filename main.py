"""Entry point for pdf-text-extraction.

Usage:
    python main.py <pdf_path>
"""
import sys

from pdf_extractor.cli import run

if __name__ == "__main__":
    sys.exit(run())
