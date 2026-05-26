#!/usr/bin/env python3
"""Export saved inline images from an executed Jupyter notebook.

This is useful when a notebook has already been run and contains rendered
matplotlib outputs. It extracts image/png, image/jpeg, image/svg+xml, and
application/pdf outputs into a folder next to the notebook and writes a manifest.

Example:
    python scripts/export_notebook_images.py notebooks/aog_hkg_diagnostics.ipynb
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "notebook"


def _as_text(payload: Any) -> str:
    if isinstance(payload, list):
        return "".join(str(x) for x in payload)
    return str(payload)


def _decode_binary_payload(payload: Any) -> bytes:
    text = _as_text(payload).strip()
    return base64.b64decode(text)


def _iter_rich_outputs(nb: dict[str, Any]) -> Iterable[tuple[int, int, str, Any]]:
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        for output_idx, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {}) or {}
            for mime, payload in data.items():
                if mime in {"image/png", "image/jpeg", "image/svg+xml", "application/pdf"}:
                    yield cell_idx, output_idx, mime, payload


def export_notebook_images(notebook: Path, out_dir: Path | None = None, zip_outputs: bool = True) -> tuple[Path, Path | None, int]:
    notebook = notebook.expanduser().resolve()
    if out_dir is None:
        out_dir = notebook.with_suffix("").parent / f"{notebook.stem}_images"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with notebook.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    suffix_for_mime = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/svg+xml": ".svg",
        "application/pdf": ".pdf",
    }
    rows: list[dict[str, Any]] = []
    count = 0
    for cell_idx, output_idx, mime, payload in _iter_rich_outputs(nb):
        count += 1
        suffix = suffix_for_mime[mime]
        file_name = f"{count:03d}_cell{cell_idx:03d}_output{output_idx:02d}{suffix}"
        path = out_dir / file_name
        if mime in {"image/png", "image/jpeg", "application/pdf"}:
            path.write_bytes(_decode_binary_payload(payload))
        else:
            path.write_text(_as_text(payload), encoding="utf-8")
        rows.append({
            "idx": count,
            "cell_idx": cell_idx,
            "output_idx": output_idx,
            "mime": mime,
            "filename": file_name,
            "path": str(path),
        })

    manifest = out_dir / "notebook_image_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "cell_idx", "output_idx", "mime", "filename", "path"])
        writer.writeheader()
        writer.writerows(rows)

    zip_path: Path | None = None
    if zip_outputs:
        zip_path = out_dir.parent / f"{out_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(out_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(out_dir.parent))
    return out_dir, zip_path, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inline notebook images to local files.")
    parser.add_argument("notebook", type=Path, help="Path to an executed .ipynb file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to <notebook_stem>_images next to the notebook.")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a zip archive of exported images.")
    args = parser.parse_args()

    out_dir, zip_path, count = export_notebook_images(args.notebook, args.out_dir, zip_outputs=not args.no_zip)
    print(f"Exported {count} image outputs to {out_dir}")
    if zip_path is not None:
        print(f"Created zip archive: {zip_path}")
    if count == 0:
        print("No inline image outputs were found. Re-run the notebook with SAVE_FIGURES=True to generate PNG files directly.")


if __name__ == "__main__":
    main()
