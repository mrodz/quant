from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Axis:
    id: str
    side: Literal["left", "right"] = "left"


@dataclass
class SeriesGroup:
    """
    A group of aligned bond spread series with an optional overlay series
    (e.g. equity), providing cross-sectional mean/std plotting.

    Parameters
    ----------
    series  : list of pd.Series — the primary series (e.g. Z-spreads per bond)
    z       : std multiplier for the shaded band (default 0.25)
    overlay : optional secondary Series plotted on a twin y-axis
    axis    : optional Axis — controls which axis this group is plotted on
              in a SeriesGroupStack (side="left" or side="right")
    labels  : optional dict with keys "primary", "overlay", "title",
              "ylabel_primary", "ylabel_overlay"
    """
    series: list[pd.Series]
    z: Optional[float] = 0.25
    overlay: Optional[pd.Series] = None
    axis: Optional[Axis] = None
    labels: dict = field(default_factory=dict)

    # ── cosmetics ────────────────────────────────────────────────────────────
    PRIMARY_COLOR: str = "#d85a30"
    OVERLAY_COLOR: str = "#283F71"

    # ── internal helpers ─────────────────────────────────────────────────────

    def _cleaned_df(self) -> pd.DataFrame:
        """
        Align all series into a DataFrame, then strip the first/last
        non-NaN value of each column (workaround for the upstream library bug
        that contaminates boundary rows).
        """
        df = pd.concat(self.series, axis=1).sort_index()

        for col in df.columns:
            s = df[col].sort_index(ascending=False).dropna()
            if s.empty:
                continue
            val_first, val_last = s.iloc[0], s.iloc[-1]
            df[col] = df.loc[
                (df[col] != val_first) & (df[col] != val_last), col
            ]

        return df

    def _mean_std(self) -> tuple[pd.Series, pd.Series]:
        df = self._cleaned_df()
        mean = df.mean(axis=1).sort_index().dropna()
        std  = df.std(axis=1).sort_index()
        return mean, std

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def is_single(self) -> bool:
        return len(self.series) == 1

    def plot(
        self,
        figsize: tuple[float, float] = (14, 8),
        ax: Optional[maxes.Axes] = None,
        color: Optional[str] = None,
    ) -> tuple[Figure, maxes.Axes]:
        """
        Draw the cross-sectional mean ± z*SD band, and optionally an overlay
        series on a twin axis.

        Returns (fig, ax) — or (fig, (ax, ax2)) when an overlay is present.
        """
        mean, std = self._mean_std()
        primary_color = color or self.PRIMARY_COLOR

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # ── primary: mean line ───────────────────────────────────────────────
        primary_label = self.labels.get("primary", None)
        ax.plot(
            mean.index, mean.values,
            color=primary_color, linewidth=1.5,
            label=primary_label,
        )

        # ── ±z·SD band — skipped for single series ───────────────────────────
        band_label = None
        if not self.is_single and std is not None and self.z is not None:
            lower = (mean - self.z * std).groupby(level=0).mean()
            upper = (mean + self.z * std).groupby(level=0).mean()
            band_label = f"±{self.z} SD"

            ax.fill_between(
                mean.index,
                lower.reindex(mean.index),
                upper.reindex(mean.index),
                color=primary_color, alpha=0.2,
                label=band_label,
            )

        if (ylabel_primary := self.labels.get("ylabel_primary", None)) is not None:
            ax.set_ylabel(ylabel_primary, color="#181717", fontsize=10)

        ax.tick_params(axis="y")
        ax.set_xlabel("Date")

        if (title := self.labels.get("title", None)) is not None:
            ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

        # ── overlay on twin axis ─────────────────────────────────────────────
        if self.overlay is not None:
            ax2 = ax.twinx()
            eq = self.overlay.sort_index().dropna()
            overlay_label = self.labels.get("overlay", self.overlay.name or "Overlay")

            ax2.plot(
                eq.index, eq.values,
                color=self.OVERLAY_COLOR, linewidth=1.2, alpha=0.6,
                label=overlay_label,
            )

            ylabel_overlay = self.labels.get("ylabel_overlay", overlay_label)
            ax2.set_ylabel(ylabel_overlay, color=self.OVERLAY_COLOR, fontsize=10)
            ax2.tick_params(axis="y", labelcolor=self.OVERLAY_COLOR)

            # unified legend
            lines  = ax.get_lines() + ([ax.collections[0]] if band_label else []) + ax2.get_lines()
            labels = (
                [l.get_label() for l in ax.get_lines()]
                + ([band_label] if band_label else [])
                + [l.get_label() for l in ax2.get_lines()]
            )
            ax.legend(lines, labels, fontsize=9, loc="upper left")

            return fig, (ax, ax2)

        ax.legend(fontsize=9, loc="upper left")
        return fig, ax


@dataclass
class SeriesGroupStack:
    """
    Plot multiple SeriesGroups on a shared figure, routing each group to a
    left or right y-axis based on its ``axis.side`` value.

    Groups sharing the same ``Axis`` instance (or the same ``axis.id``) are
    drawn on the same matplotlib Axes object, so their scales are shared.
    Groups with no ``axis`` assigned default to the left axis.

    Parameters
    ----------
    groups : list of SeriesGroup
    colors : colour cycle applied to groups in order
    """
    groups: list[SeriesGroup]
    colors: list[str] = field(default_factory=lambda: [
        "#d85a30", "#283F71", "#2a9d8f", "#e9c46a",
    ])

    def plot(
        self,
        figsize: tuple[float, float] = (14, 8),
        title: str = "",
    ) -> tuple[Figure, maxes.Axes | tuple[maxes.Axes, maxes.Axes]]:
        """
        Draw all groups.

        Returns
        -------
        (fig, ax_left)                     — if no group uses side="right"
        (fig, (ax_left, ax_right))         — if at least one group uses side="right"
        """
        fig, ax_left = plt.subplots(figsize=figsize)
        ax_right: Optional[maxes.Axes] = None

        # track which axis ids have already had a ylabel set
        _ylabel_set: dict[str, bool] = {}

        for i, group in enumerate(self.groups):
            color = self.colors[i % len(self.colors)]
            mean, std = group._mean_std()
            label = group.labels.get("title", f"Group {i}")

            # ── axis routing ─────────────────────────────────────────────────
            side = group.axis.side if group.axis is not None else "left"

            if side == "right":
                if ax_right is None:
                    ax_right = ax_left.twinx()
                target_ax = ax_right
            else:
                target_ax = ax_left

            # ── mean line ────────────────────────────────────────────────────
            target_ax.plot(
                mean.index, mean.values,
                color=color, linewidth=1.5, label=label,
            )

            # ── ±z·SD band ───────────────────────────────────────────────────
            if not group.is_single and std is not None and group.z is not None:
                lower = (mean - group.z * std).groupby(level=0).mean()
                upper = (mean + group.z * std).groupby(level=0).mean()
                target_ax.fill_between(
                    mean.index,
                    lower.reindex(mean.index),
                    upper.reindex(mean.index),
                    color=color, alpha=0.15,
                    label=f"{label} ±{group.z} SD",
                )

            # ── per-axis ylabel (first group to claim the axis wins) ─────────
            axis_key = (group.axis.id if group.axis else side)
            if axis_key not in _ylabel_set:
                ylabel = group.labels.get("ylabel_primary", "")
                if ylabel:
                    target_ax.set_ylabel(ylabel, fontsize=10, color=color)
                    if side == "right" and ax_right is not None:
                        ax_right.tick_params(axis="y", labelcolor=color)
                _ylabel_set[axis_key] = True

        # ── shared decoration ────────────────────────────────────────────────
        ax_left.set_xlabel("Date")
        ax_left.set_title(title, fontsize=12)
        ax_left.grid(True, alpha=0.3)

        # unified legend: collect from both axes
        all_handles, all_labels = ax_left.get_legend_handles_labels()
        if ax_right is not None:
            r_handles, r_labels = ax_right.get_legend_handles_labels()
            all_handles += r_handles
            all_labels  += r_labels

        ax_left.legend(all_handles, all_labels, fontsize=9, loc="upper left")

        if ax_right is not None:
            return fig, (ax_left, ax_right)
        return fig, ax_left