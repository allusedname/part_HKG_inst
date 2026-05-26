# AOG-HKG diagnostic figure export

The diagnostic notebook now saves every major matplotlib figure as a local PNG while still showing it inline.

Default output location:

```text
notebooks/aog_hkg_diagnostic_images/
```

At the end of the notebook, `finalize_figure_exports()` writes:

```text
notebooks/aog_hkg_diagnostic_images/figure_manifest.csv
notebooks/aog_hkg_diagnostic_images/figure_manifest.json
notebooks/aog_hkg_diagnostic_images.zip
```

The zip file is the easiest artifact to upload or share.

## Settings

In the notebook configuration cell:

```python
SAVE_FIGURES = True
FIGURE_DPI = 180
SAVE_PDF_FIGURES = False
FIGURE_DIR = NOTEBOOK_DIR / "aog_hkg_diagnostic_images"
```

Set `SAVE_PDF_FIGURES=True` to save vector-style PDF copies in addition to PNGs.

## Export already-rendered notebook outputs

If you already have an executed notebook with inline images saved in its outputs, run:

```bash
python scripts/export_notebook_images.py notebooks/aog_hkg_diagnostics.ipynb
```

This extracts any embedded image outputs into:

```text
notebooks/aog_hkg_diagnostics_images/
```

and creates:

```text
notebooks/aog_hkg_diagnostics_images.zip
```

If the notebook was saved without outputs, the script will report zero images; in that case, re-run the notebook with `SAVE_FIGURES=True`.
