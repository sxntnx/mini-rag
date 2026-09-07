"""CLI de ingesta. Uso: python ingest.py [carpeta]"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from rag import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_docs")
    start = time.perf_counter()
    stats = RagPipeline().ingest_folder(folder)
    stats["elapsed_s"] = round(time.perf_counter() - start, 2)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
