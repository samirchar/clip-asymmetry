"""Shared figure infrastructure for the paper plots.

The single source of truth for figure style (the frozen :class:`Theme`), the
per-encoder colour palette construction, small formatting helpers, and artifact
saving. Imported by ``bin/generate_figures.py`` (the figure builders) and by
``src.analysis.load_data`` (which attaches the palettes to the loaded data).
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
PAPER_DIR = ROOT.parent / "clip_research"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"


@dataclass(frozen=True)
class Theme:
    """Global figure style. One source of truth for fonts, sizes, line widths,
    marker sizes, palette config, and accent colours. Override per-figure with
    ``dataclasses.replace(THEME, ...)`` and apply via ``mpl.rc_context(theme.rc())``
    (figure-specific *layout* — annotation offsets, axis limits — stays local).
    """
    # Canvas — TMLR \textwidth = 6.5in (single column, 10pt article)
    textwidth: float = 6.5
    # Font family — match the TMLR body (Computer Modern / Latin Modern serif)
    font_family: str = "serif"
    serif_fonts: tuple = ("CMU Serif", "cmr10", "Latin Modern Roman", "DejaVu Serif")
    sans_fonts: tuple = ("Arial", "DejaVu Sans", "Helvetica")
    mathtext: str = "cm"
    # Font sizes (pt)
    fs_base: float = 9
    fs_title: float = 9
    fs_label: float = 9
    fs_tick: float = 8
    fs_legend: float = 7
    fs_legend_title: float = 8
    fs_panel: float = 9          # (a)/(b)/(c) panel labels
    fs_annot: float = 7          # default in-plot annotation
    fs_annot_sm: float = 6.5     # dense panels
    fs_annot_xs: float = 6       # very dense panels
    # Line widths / marker sizes
    lw_axis: float = 0.5
    lw_grid: float = 0.3
    lw_plot: float = 0.9
    lw_arrow: float = 1.4
    marker_size: float = 3.5     # plot() markersize (points)
    scatter_area: float = 26     # scatter() s (points^2)
    # DPI
    dpi: int = 300
    # Palette: encoder shade ramps + single modality accents
    vision_cmap: str = "Blues"
    text_cmap: str = "RdPu"
    cmap_lo: float = 0.20        # low end of the encoder shade ramp
    cmap_hi: float = 0.98        # high end
    vision_accent_pos: float = 0.72
    text_accent_pos: float = 0.75
    grey: str = "#333333"
    grey_light: str = "#666666"
    grey_mid: str = "0.55"

    def rc(self):
        """matplotlib rcParams for this theme."""
        if self.font_family == "serif":
            family, fam_key, fam = "serif", "font.serif", self.serif_fonts
        else:
            family, fam_key, fam = "sans-serif", "font.sans-serif", self.sans_fonts
        return {
            "font.family": family,
            fam_key: list(fam),
            "font.size": self.fs_base,
            "axes.labelsize": self.fs_label,
            "axes.titlesize": self.fs_title,
            "xtick.labelsize": self.fs_tick,
            "ytick.labelsize": self.fs_tick,
            "legend.fontsize": self.fs_legend,
            "legend.title_fontsize": self.fs_legend_title,
            "mathtext.fontset": self.mathtext,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "lines.linewidth": self.lw_plot,
            "lines.markersize": self.marker_size,
            "axes.linewidth": self.lw_axis,
            "grid.linewidth": self.lw_grid,
            "grid.alpha": 0.3,
        }

    def vision_accent(self):
        return mpl.colors.to_hex(plt.get_cmap(self.vision_cmap)(self.vision_accent_pos))

    def text_accent(self):
        return mpl.colors.to_hex(plt.get_cmap(self.text_cmap)(self.text_accent_pos))

    def size(self, rel_w=1.0, h=3.5):
        """figsize helper: width = rel_w * textwidth (so width=\\linewidth places 1:1)."""
        return (self.textwidth * rel_w, h)


THEME = Theme()  # TMLR: 6.5in textwidth, Computer Modern serif


TEXTWIDTH = THEME.textwidth  # TMLR \textwidth (6.5in); use for figsize widths


SIZES = {
    "full": (THEME.textwidth, 3.5),
    "full_tall": (THEME.textwidth, 5.0),
    "full_wide": (THEME.textwidth, 2.8),
    "half": (THEME.textwidth / 2, 2.8),
    "si": (4.17, 3.5),
}


DATASET_PALETTE = {"cc3m": "#2ca02c", "cc12m": "#d62728"}


def _build_palette_and_labels(arch_df, col_name, col_params, cmap_name):
    """Build palette and labels dicts from architectures dataframe.

    Keys are param counts. Colors are sampled from a sequential colormap
    evenly spaced by rank (not by param value), so all entries are visually
    distinguishable regardless of how close their param counts are.
    """
    unique = (
        arch_df[[col_name, col_params]]
        .drop_duplicates(subset=[col_name])
        .sort_values(col_params)
    )
    params_list = unique[col_params].tolist()
    names_list = unique[col_name].tolist()
    n = len(params_list)

    cmap = plt.get_cmap(cmap_name)
    # Evenly space colors in the visible range of the colormap (theme-controlled)
    positions = np.linspace(THEME.cmap_lo, THEME.cmap_hi, n)
    colors = [cmap(p) for p in positions]

    palette = dict(zip(params_list, [mpl.colors.to_hex(c) for c in colors]))
    labels = {
        p: f"{name} ({format_params(p)})"
        for p, name in zip(params_list, names_list)
    }
    return palette, labels


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.0)
    mpl.rcParams.update(THEME.rc())


def label_panels(axes, theme=THEME):
    """Add (a), (b), (c), ... labels to a flat sequence of axes.

    Replaces any existing center title with just the panel letter.
    """
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    flat = np.array(axes).flat
    for i, ax in enumerate(flat):
        if ax.get_visible():
            ax.set_title("", loc="center")
            ax.set_title(
                f"({chr(ord('a') + i)})",
                fontsize=theme.fs_panel, fontweight="bold", loc="left",
            )


def format_params(val):
    """Format parameter count for legend labels (e.g., '86M')."""
    if val >= 1e6:
        return f"{val/1e6:.0f}M"
    elif val >= 1e3:
        return f"{val/1e3:.0f}K"
    return str(int(val))


def format_params_axis(val):
    """Format parameter count for axis ticks (millions, no suffix)."""
    return f"{val/1e6:.0f}"


def millions_formatter(suffix=""):
    """Axis tick formatter: parameter counts in millions (e.g. ``86`` or ``86M``)."""
    return mpl.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}{suffix}")


def save(fig, name):
    """Save figure as PDF and SVG to CLIPAblation/figures/ and copy to clip_research/figures/."""
    import shutil

    out_dir = FIGURES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / f"{name}.pdf", out_dir / f"{name}.svg"]
    for path in written:
        fig.savefig(path)

    # Mirror ONLY the artifacts written this run to clip_research/figures/ (do not
    # copy stale/manual variants left in the directory, e.g. hand-edited *_v2.pdf).
    paper_out = PAPER_FIGURES_DIR / name
    if PAPER_FIGURES_DIR.parent.exists():
        paper_out.mkdir(parents=True, exist_ok=True)
        for path in written:
            shutil.copy2(path, paper_out / path.name)
