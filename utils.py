import h5py as h5


def pack_hdf(pth, data_dict):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure."""

    def rec(data, grp):
        for k, v in data.items():
            if type(v) is dict:
                rec(v, grp.create_group(k))
            else:
                grp.create_dataset(k, data=v)

    with h5.File(pth + ".h5", "w") as pckg:
        rec(data_dict, pckg)


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        k: v[()] if type(v) is h5._hl.dataset.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }


def clean_axes(
    axes,
    remove_spines=["right", "top"],
    ticksize=11.0,
    spine_width=None,
    tick_width=None,
):
    """A couple basic changes I often make to pyplot axes. If input is an
    iterable of axes (e.g. from plt.subplots()), apply recursively."""
    if hasattr(axes, "__iter__"):
        for a in axes:
            clean_axes(
                a,
                remove_spines=remove_spines,
                ticksize=ticksize,
                tick_width=tick_width,
                spine_width=spine_width,
            )
    else:
        if spine_width is not None and tick_width is None:
            tick_width = spine_width
        for r in remove_spines:
            axes.spines[r].set_visible(False)
        axes.tick_params(
            axis="both",
            which="both",  # applies to major and minor
            **{r: False for r in remove_spines},  # remove ticks
            **{"label%s" % r: False for r in remove_spines},  # remove labels
            width=tick_width,
        )
        for ticks in axes.get_yticklabels():
            ticks.set_fontsize(ticksize)
        for ticks in axes.get_xticklabels():
            ticks.set_fontsize(ticksize)
        if spine_width is not None:
            for s in ["top", "bottom", "left", "right"]:
                axes.spines[s].set_linewidth(spine_width)


def dummy_yaxis_label(ax, lbl, offset=-0.6, **ylabel_kwargs):
    """Create a dummy twin axis so that an additional (usually horizontal) label
    can be added to the left of the actual ylabel. Useful for labelling rows of
    subplots."""
    if "rotation" not in ylabel_kwargs:
        ylabel_kwargs["rotation"] = 0
    lbl_ax = ax.twinx()
    lbl_ax.yaxis.set_label_position("left")
    lbl_ax.spines["left"].set_position(("axes", offset))
    lbl_ax.spines["left"].set_visible(False)
    lbl_ax.spines["top"].set_visible(False)
    lbl_ax.spines["right"].set_visible(False)
    lbl_ax.set_yticks([])
    lbl_ax.set_ylabel(lbl, size="large", ha="right", va="center", **ylabel_kwargs)
    return lbl_ax
