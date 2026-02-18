"""
HRPUB-Compliant Figure Manager
================================

Automates the saving of matplotlib figures following HRPUB manuscript standards:
  - Sequential naming: Figure_1.png, Figure_2.png, ...
  - 300 DPI minimum resolution
  - Consistent sizing for journal submission
  - Automatic caption registry (exportable to Markdown/Word)

Usage:
    from src.utils.figure_manager import FigureManager

    fm = FigureManager(output_dir="figures")

    fig, ax = plt.subplots()
    ax.plot(time, drift)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inter-story Drift Ratio")

    fm.save(fig, caption="Inter-story drift response for RSN953 Northridge record.")

Author: Mikisbell
Project: Hybrid Digital Twin for Seismic RC Buildings
"""

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FigureManager:
    """
    Manages figure saving with HRPUB-compliant naming and resolution.

    Parameters
    ----------
    output_dir : str or Path
        Directory where figures are saved.
    dpi : int
        Resolution in dots per inch (HRPUB requires >= 300).
    fmt : str
        Image format: 'png', 'pdf', 'eps', 'svg', 'tiff'.
    figsize : tuple
        Default figure size in inches (width, height).
        HRPUB single-column: (3.5, 2.8), double-column: (7.0, 4.5).
    font_size : int
        Base font size for all text elements.
    """

    # HRPUB recommended figure widths (inches)
    SINGLE_COLUMN = (3.5, 2.8)
    DOUBLE_COLUMN = (7.0, 4.5)
    FULL_PAGE = (7.0, 9.0)

    def __init__(
        self,
        output_dir: str = "figures",
        dpi: int = 300,
        fmt: str = "png",
        figsize: tuple = (7.0, 4.5),
        font_size: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = max(dpi, 300)  # Enforce HRPUB minimum
        self.fmt = fmt
        self.figsize = figsize
        self.font_size = font_size
        self._counter = self._detect_next_index()
        self._captions: list[dict] = self._load_caption_registry()

        # Apply consistent style for all figures
        self._apply_journal_style()

        logger.info(
            f"FigureManager initialized: dir={self.output_dir}, "
            f"next_index={self._counter}, dpi={self.dpi}"
        )

    # -------------------------------------------------------------------
    # Core: Save a figure
    # -------------------------------------------------------------------
    def save(
        self,
        fig: matplotlib.figure.Figure,
        caption: str = "",
        label: str | None = None,
        tight: bool = True,
    ) -> Path:
        """
        Save a matplotlib figure with HRPUB-compliant naming.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        caption : str
            Figure caption for the manuscript.
        label : str, optional
            Custom label (e.g., "drift_response"). If None, uses Figure_N.
        tight : bool
            Apply tight_layout before saving.

        Returns
        -------
        Path
            Path to the saved figure file.
        """
        if tight:
            fig.tight_layout()

        # Build filename
        filename = f"{label}.{self.fmt}" if label else f"Figure_{self._counter}.{self.fmt}"

        filepath = self.output_dir / filename

        fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.05,
        )

        # Register caption
        entry = {
            "figure": filename,
            "number": self._counter,
            "caption": caption,
            "label": label or f"fig:{self._counter}",
            "path": str(filepath),
        }
        self._captions.append(entry)
        self._save_caption_registry()

        logger.info(f"📊 Saved: {filepath} ({self.dpi} DPI)")

        self._counter += 1
        return filepath

    # -------------------------------------------------------------------
    # Export captions for manuscript
    # -------------------------------------------------------------------
    def export_captions_markdown(self) -> str:
        """
        Export all figure captions as Markdown text,
        ready to paste into the HRPUB manuscript.
        """
        lines = ["## List of Figures\n"]
        for entry in self._captions:
            n = entry["number"]
            cap = entry["caption"]
            lines.append(f"**Figure {n}.** {cap}\n")

        md_text = "\n".join(lines)

        # Also save to file
        md_path = self.output_dir / "figure_captions.md"
        md_path.write_text(md_text, encoding="utf-8")
        logger.info(f"Caption list exported to {md_path}")

        return md_text

    def export_captions_plain(self) -> str:
        """
        Export captions as plain text for Word template insertion.
        """
        lines = []
        for entry in self._captions:
            n = entry["number"]
            cap = entry["caption"]
            lines.append(f"Figure {n}. {cap}")
        return "\n\n".join(lines)

    # -------------------------------------------------------------------
    # Style configuration (HRPUB-compliant)
    # -------------------------------------------------------------------
    def _apply_journal_style(self):
        """Set matplotlib rcParams for journal-quality figures."""
        plt.rcParams.update(
            {
                "figure.figsize": self.figsize,
                "figure.dpi": self.dpi,
                "font.size": self.font_size,
                "font.family": "serif",
                "axes.labelsize": self.font_size,
                "axes.titlesize": self.font_size + 1,
                "xtick.labelsize": self.font_size - 1,
                "ytick.labelsize": self.font_size - 1,
                "legend.fontsize": self.font_size - 1,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.2,
                "lines.markersize": 4,
                "savefig.dpi": self.dpi,
                "savefig.bbox": "tight",
                "savefig.facecolor": "white",
            }
        )

    # -------------------------------------------------------------------
    # Internal: auto-detect next figure index
    # -------------------------------------------------------------------
    def _detect_next_index(self) -> int:
        """Scan output_dir for existing Figure_N files and return next index."""
        existing = list(self.output_dir.glob(f"Figure_*.{self.fmt}"))
        if not existing:
            return 1
        indices = []
        for f in existing:
            try:
                idx = int(f.stem.split("_")[1])
                indices.append(idx)
            except (ValueError, IndexError):
                continue
        return max(indices, default=0) + 1

    # -------------------------------------------------------------------
    # Internal: caption registry persistence
    # -------------------------------------------------------------------
    def _load_caption_registry(self) -> list[dict]:
        registry_path = self.output_dir / "caption_registry.json"
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_caption_registry(self):
        registry_path = self.output_dir / "caption_registry.json"
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(self._captions, f, indent=2, ensure_ascii=False)
