import os
# import h5py as h5

import numpy as np
import matplotlib.pyplot as plt

from diffusion import disc2D, space3D, ach_2D, ach_3D, glut_2D, glut_3D
import graph_builds as gb
import utils


def total_open(state_recs):
    """Take state recording dict and return total value on graph residing
    in open states, denoted by '*' the keys."""
    return sum([state_recs[s] for s in state_recs.keys() if "*" in s])


def diffusion2D_alpha_comparison(save_pth=None, fmt="png"):
    """Side-by side comparison of alpha7 and alpha3 responses to 2D diffusion
    based agonist stimulation profiles (also plotted)."""
    a7_builder = gb.loader(gb.alpha7)
    a3_builder = gb.loader(gb.alpha3)
    prox_func = ach_2D(0.0)
    distal_func = ach_2D(1.1e-6)

    a7_prox = a7_builder(agonist_func=prox_func)
    a7_distal = a7_builder(agonist_func=distal_func)
    a3_prox = a3_builder(agonist_func=prox_func)
    a3_distal = a3_builder(agonist_func=distal_func)

    a7_open_prox = a7_prox.run()["A2R*"]
    a7_open_distal = a7_distal.run()["A2R*"]
    a3_open_prox = a3_prox.run()["A2R*"]
    a3_open_distal = a3_distal.run()["A2R*"]

    fig, ax = plt.subplots(3, sharex=True, figsize=(5, 8))

    t = a3_prox.time / 1000
    ax[0].plot(a3_prox.time, prox_func(t) / 1000, label="Proximal (r = 0)")
    ax[0].plot(a3_prox.time, distal_func(t) / 1000, label="Distal (r = 1.1μm)")
    ax[0].set_ylabel("Concentration (M)", fontsize=12)
    ax[0].set_yscale("log")
    ax[0].set_ylim(5e-7, 1e-3)
    ax[0].set_title("2D Disc Diffusion")

    ax[1].plot(a7_prox.time, a7_open_prox, label="Proximal (r = 0)")
    ax[1].plot(a7_distal.time, a7_open_distal, label="Distal (r = 1.1μm)")
    ax[1].set_ylabel("Open Probability", fontsize=12)
    ax[1].set_title("Alpha 7")

    ax[2].plot(a3_prox.time, a3_open_prox, label="Proximal (r = 0)")
    ax[2].plot(a3_distal.time, a3_open_distal, label="Distal (r = 1.1μm)")
    ax[2].set_xlim(0, 25)
    ax[2].set_xlabel("Time (ms)", fontsize=12)
    ax[2].set_ylabel("Open Probability", fontsize=12)
    ax[2].set_title("Alpha 3")

    for a in ax:
        for ticks in (a.get_xticklabels() + a.get_yticklabels()):
            ticks.set_fontsize(11)
        a.legend(frameon=False, fontsize=11)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    fig.tight_layout()

    if save_pth is not None:
        fname = "nACHR_2D_diffusion_comparison.%s" % fmt
        fig.savefig(save_pth + fname, bbox_inches="tight")

    plt.show()


def alpha7_vs_alpha6(save_pth=None, fmt="png"):
    a7_builder = gb.loader(gb.alpha7)
    a6_builder = gb.loader(gb.alpha7, desens_div=4)
    prox_func = ach_2D(0.0)
    distal_func = ach_2D(1.1e-6)

    a7_prox = a7_builder(agonist_func=prox_func)
    a7_distal = a7_builder(agonist_func=distal_func)
    a6_prox = a6_builder(agonist_func=prox_func)
    a6_distal = a6_builder(agonist_func=distal_func)

    a7_open_prox = a7_prox.run()["A2R*"]
    a7_open_distal = a7_distal.run()["A2R*"]
    a6_open_prox = a6_prox.run()["A2R*"]
    a6_open_distal = a6_distal.run()["A2R*"]

    fig, ax = plt.subplots(1)

    t = a6_prox.time / 1000
    ax.plot(a7_prox.time, a7_open_prox, c="C0", label="alpha7 (r = 0)")
    ax.plot(a7_distal.time, a7_open_distal, c="C1", label="alpha7 (r = 1.1μm)")
    ax.plot(a6_prox.time, a6_open_prox, c="C0", label="alpha6 (r = 0)", linestyle="--")
    ax.plot(a6_distal.time, a6_open_distal, c="C1", label="alpha6 (r = 1.1μm)", linestyle="--")
    ax.set_xlim(0, 25)

    ax.set_ylabel("Open Probability", fontsize=12)
    ax.set_xlabel("Time (ms)", fontsize=12)

    for ticks in (ax.get_xticklabels() + ax.get_yticklabels()):
        ticks.set_fontsize(11)
    ax.legend(frameon=False, fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()

    if save_pth is not None:
        fname = "alpha7_vs_alpha6.%s" % fmt
        fig.savefig(save_pth + fname, bbox_inches="tight")

    plt.show()


def gaba_paper_mimic():
    model = gb.loader(gb.GABA)(pulses=[(0, 2, 10)])
    control = total_open(model.run())
    model.update_edge("A2R", "A2D", 2.15)
    nocodazole = total_open(model.run())

    fig, ax = plt.subplots(1, figsize=(3., 3.5))
    ax.plot(model.time, control, label="Control")
    ax.plot(model.time, nocodazole, label="Nocodazole")
    ax.set_xlim(-1, 20)
    ax.set_ylim(0, .6)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Open Probability", fontsize=12)
    ax.set_title("GABAR; 2ms pulse @ 10mM")

    for ticks in (ax.get_xticklabels() + ax.get_yticklabels()):
        ticks.set_fontsize(11)
    ax.legend(frameon=False, fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    plt.show()


def updated_diffusion():
    time_ax = np.arange(1, 25001) * .001  # [ms]
    time = time_ax / 1000  # [s]

    centre_og = disc2D(4700, 7.6e-10, 20e-9, 0.)(time) / 1000
    dist_og = disc2D(4700, 7.6e-10, 20e-9, 1.1e-6)(time) / 1000

    centre_new = disc2D(10000, 4e-10, 20e-9, 0.)(time) / 1000
    dist_new = disc2D(10000, 4e-10, 20e-9, 1.1e-6)(time) / 1000

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.plot(time_ax, centre_og, c="k", label="r = 0")
    ax.plot(time_ax, dist_og, c=".5", label="r = 1.1μm")
    ax.plot(time_ax, centre_new, c="k", linestyle="--", label="r = 0 (ACH)")
    ax.plot(time_ax, dist_new, c=".5", linestyle="--", label="r = 1.1μm (ACH)")
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 1e-3)
    ax.set_xlim(0, 5)
    ax.set_ylabel("Concentration (M)", fontsize=12)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.legend(frameon=False, fontsize=11)

    for ticks in (ax.get_xticklabels() + ax.get_yticklabels()):
        ticks.set_fontsize(11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()


def pulse_plot(graph_builder, pulses, title=""):
    """Load kinetic graph up with agonist pulses, run it and plot open
    probability over time."""
    model = graph_builder(pulses=pulses)
    recs = model.run()
    open_pop = total_open(recs)

    fig, ax = plt.subplots(1)
    ax.plot(model.time, open_pop)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Open Probability", fontsize=12)
    ax.set_title(title)

    fig.tight_layout()
    plt.show()


def prox_vs_distal(graph_builder, threeD=False, trans="ach", title=None,
                   save_pth=None, fmt="png"):
    """Load kinetic graph with proximal and distal 2D diffusion agonist
    profiles, then run it and plot open probability over time for each for
    comparison."""
    if not threeD:
        d_flag = "2D"
        if trans == "ach":
            prox_func = ach_2D(0)
            distal_func = ach_2D(1.1e-6)
        else:
            prox_func = glut_2D(0)
            distal_func = glut_2D(1.1e-6)
    else:
        d_flag = "3D"
        if trans == "ach":
            prox_func = ach_3D(0)
            distal_func = ach_3D(1.1e-6)
        else:
            prox_func = glut_3D(0)
            distal_func = glut_3D(1.1e-6)

    model_prox = graph_builder(agonist_func=prox_func)
    model_distal = graph_builder(agonist_func=distal_func)

    open_prox = total_open(model_prox.run())
    open_distal = total_open(model_distal.run())

    print("Peak ratio (distal / prox): %.3f" % (open_distal.max() / open_prox.max()))

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.plot(model_prox.time, open_prox, label="Proximal (r = 0)")
    ax.plot(model_distal.time, open_distal, label="Distal (r = 1.1μm)")
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Open Probability", fontsize=12)

    if title is None:
        title = "%s (Space %s)" % (model_prox.name, d_flag)
    ax.set_title(title)

    for ticks in (ax.get_xticklabels() + ax.get_yticklabels()):
        ticks.set_fontsize(11)
    ax.legend(frameon=False, fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()

    if save_pth is not None:
        fname = "prox_vs_distal_%s_%s%s.%s" % (model_prox.name, trans, d_flag, fmt)
        fig.savefig(save_pth + fname, bbox_inches="tight")

    plt.show()


def prox_vs_distal_states(graph_builder):
    """Load kinetic graph with proximal and distal 2D diffusion agonist
    profiles, then run it and plot state values of time for both."""
    # prox_func = disc2D(4700, 7.6e-10, 20e-9, 0)
    # distal_func = disc2D(4700, 7.6e-10, 20e-9, 1.1e-6)
    prox_func = disc2D(10000, 4e-10, 20e-9, 0)
    distal_func = disc2D(10000, 4e-10, 20e-9, 1.1e-6)

    models = {
        "Proximal": graph_builder(agonist_func=prox_func),
        "Distal": graph_builder(agonist_func=distal_func)
    }
    recs = {k: model.run() for k, model in models.items()}

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
    for (name, model), rec, a in zip(models.items(), recs.values(), ax):
        for label, state in rec.items():
            a.plot(model.time, state, label=label)
        a.set_ylim(0, 1)
        a.set_xlabel("Time (ms)", fontsize=12)
        a.set_title(name, fontsize=14)

        for ticks in (a.get_xticklabels() + a.get_yticklabels()):
            ticks.set_fontsize(11)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    ax[0].legend(frameon=False, fontsize=11)
    ax[0].set_ylabel("Probability", fontsize=12)

    fig.tight_layout()
    plt.show()


def rate_modulation(builder, edges, agonist_funcs=None, mul_range=10, plot=True):
    """Run series of experiments with the rates of given edges multiplied by
    factors from (1 / mul_range to mul_range) for a set of models (e.g. proximal
    and distal diffusion fed). Edges are modified in paralell with the same
    multiplier in each experiment. Return multipliers used, open probability
    over time, and calculated descriptive metrics.
    """
    if agonist_funcs is None:
        agonists_funcs = {
            "prox": disc2D(4700, 7.6e-10, 20e-9, 0),
            "distal": disc2D(4700, 7.6e-10, 20e-9, 1.1e-6),
        }

    models = {k: builder(agonist_func=f) for k, f in agonist_funcs.items()}
    base_rates = [
        models["prox"].nodes[n1].out_edges[n2]["Rate"]
        for (n1, n2) in edges
    ]
    multis = np.array(
        [1 / i for i in range(mul_range, 0, -1)]
        + list(range(2, mul_range + 1))
    )

    recs = {k: [] for k in models.keys()}
    probs = {k: [] for k in models.keys()}
    metrics = {
        k: {"peak": [], "peak_time": [], "area": []}
        for k in models.keys()
    }

    for m in multis:
        for (n1, n2), base in zip(edges, base_rates):
            for model in models.values():
                model.update_edge(n1, n2, base * m)
        for k, model in models.items():
            rec = model.run()
            recs[k].append(rec)
            probs[k].append(total_open(rec))
            metrics[k]["peak"].append(np.max(probs[k][-1]))
            metrics[k]["peak_time"].append(
                np.where(probs[k][-1] == metrics[k]["peak"][-1])[0][0]
                * model.dt
            )
            metrics[k]["area"].append(np.sum(probs[k][-1]))

    metrics = {
        model: {metric: np.array(v2) for metric, v2 in v1.items()}
        for model, v1 in metrics.items()
    }

    if plot == True:
        edge_str = ", ".join([" → ".join(e) for e in edges])
        title = "%s; %s" % (models["prox"].name, edge_str)
        plot_modulation_metrics(multis, metrics, title)
        plot_modulation_open_prob(multis, models["prox"].time, probs, title)

    return models, multis, recs, probs, metrics


def plot_modulation_metrics(multis, metrics, title=""):
    fig, ax = plt.subplots(2, sharex=True, figsize=(5, 5))
    axr = [a.twinx() for a in ax]

    ax[0].plot(multis, metrics["prox"]["peak"], label="Proximal (0μm)")
    ax[0].plot(multis, metrics["distal"]["peak"], label="Distal (1.1μm)")
    ax[0].set_ylim(0)
    ax[0].set_ylabel("Peak Probability", fontsize=12)

    axr[0].plot(
        multis, metrics["prox"]["peak"] / metrics["distal"]["peak"],
        c="black", linestyle="--",
    )
    axr[0].set_ylim(0)
    axr[0].set_ylabel("Proximal / Distal", fontsize=12)

    ax[1].plot(multis, metrics["prox"]["peak_time"], label="Proximal (0μm)")
    ax[1].plot(multis, metrics["distal"]["peak_time"], label="Distal (1.1μm)")
    ax[1].set_ylim(0)
    ax[1].set_ylabel("Peak Time (ms)", fontsize=12)

    axr[1].plot(
        multis, metrics["distal"]["peak_time"] - metrics["prox"]["peak_time"],
        c="black", linestyle="--",
    )
    axr[1].set_ylim(0)
    axr[1].set_ylabel("Delay (ms)", fontsize=12)

    ax[-1].set_xlabel("Rate Multiplier", fontsize=12)

    for a in ax:
        for ticks in (a.get_xticklabels() + a.get_yticklabels()):
            ticks.set_fontsize(11)
        a.set_xscale("log")
        a.legend(frameon=False, fontsize=11)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95])  # rect to give space to suptitle

    return fig


def plot_modulation_open_prob(multis, time, probs, ends=True, title=""):
    """Subplots of probability over time traces, examples from half-way along
    the rate multiplier range with the un-modified case in the middle. Setting
    `ends` to True will also add the min and max of the range (making 5 plots).
    """
    l = len(multis)
    idxs = [l // 4 + 1, l // 2, l // 2 + l // 4]
    if ends:
        idxs = [0] + idxs + [len(multis) - 1]

    multis = [multis[i] for i in idxs]
    probs = {k: [v[i] for i in idxs] for k, v in probs.items()}

    fig, ax = plt.subplots(1, len(idxs), sharey=True, figsize=(8, 5))

    for a, m, prox, distal in zip(ax, multis, probs["prox"], probs["distal"]):
        a.plot(time, prox, label="Proximal (0μm)")
        a.plot(time, distal, label="Distal (1.1μm)")
        a.set_title("x %.2f" % m, fontsize=12)
        a.set_xlabel("Time (ms)", fontsize=12)
        for ticks in (a.get_xticklabels()):
            ticks.set_fontsize(11)
        a.spines["right"].set_visible(False)
        a.spines["top"].set_visible(False)

    ax[0].set_ylabel("Open Probability", fontsize=12)
    for ticks in (ax[0].get_yticklabels()):
        ticks.set_fontsize(11)

    for a in ax[1:]:
        a.spines["left"].set_visible(False)
        a.yaxis.set_ticks_position('none')

    ax[len(ax) // 2].legend(frameon=False, fontsize=11)  # middle axis

    if not ends:
        fig.tight_layout(rect=[0, 0, 1, .95])  # rect to give space to suptitle
    fig.suptitle(title, fontsize=14)

    return fig


def plot_modulation_states(multis, time, recs, ends=True, title=""):
    """Subplots of probability over time traces for each state, examples from
    half-way along the rate multiplier range with the un-modified case in the
    middle. Setting `ends` to True will also add the min and max of the range
    (making 5 plots).
    """
    l = len(multis)
    idxs = [l // 4 + 1, l // 2, l // 2 + l // 4]
    if ends:
        idxs = [0] + idxs + [len(multis) - 1]

    multis = [multis[i] for i in idxs]
    recs = {k: [v[i] for i in idxs] for k, v in recs.items()}

    fig, ax = plt.subplots(
        len(idxs), 2, sharex=True, sharey=True, figsize=(7, 10))
    # unzip axes from rows in to column-major organization
    ax = [[a[i] for a in ax] for i in range(2)]

    loc_strs = ["Proximal (0μm)", "Distal (1.1μm)"]
    for rec, col, loc in zip (recs.values(), ax, loc_strs):
        col[0].set_title(loc, fontsize=14, pad=18)
        col[-1].set_xlabel("Time (ms)", fontsize=12)
        for ticks in (col[-1].get_xticklabels()):
            ticks.set_fontsize(11)
        for m, states, a in zip(multis, rec, col):
            for label, s in states.items():
                a.plot(time, s, label=label)
            a.spines["right"].set_visible(False)
            a.spines["top"].set_visible(False)

    for m, a in zip(multis, ax[0]):
        a.set_ylabel("x %.2f" % m, fontsize=12)
        for ticks in (a.get_yticklabels()):
            ticks.set_fontsize(11)

    for a in ax[1]:
        a.spines["left"].set_visible(False)
        a.yaxis.set_ticks_position('none')

    ax[1][0].legend(
        frameon=False, loc="lower right", bbox_to_anchor=(1, .6),
        ncol=2, fontsize=11
    )

    fig.tight_layout(rect=[0, 0, 1, .95])  # rect to give space to suptitle
    fig.suptitle(title, fontsize=14)

    return fig


def binding_modulation_run(pth, builder, mul_range=10, threeD=False, trans="ach",
                           show=False, fmt="png"):
    kons = [("R", "AR"), ("AR", "A2R")]
    koffs = [("AR", "R"), ("A2R", "AR")]
    rates = {"kon": kons, "koff": koffs, "kon_koff": kons + koffs}

    agonists = {}
    if not threeD:
        d_flag = "2D"
        if trans == "ach":
            agonists["prox"] = ach_2D(0)
            agonists["distal"] = ach_2D(1.1e-6)
        else:
            agonists["prox"] = glut_2D(0)
            agonists["distal"] = glut_2D(1.1e-6)
    else:
        d_flag = "3D"
        if trans == "ach":
            agonists["prox"] = ach_3D(0)
            agonists["distal"] = ach_3D(1.1e-6)
        else:
            agonists["prox"] = glut_3D(0)
            agonists["distal"] = glut_3D(1.1e-6)

    data = {}
    for label, edges in rates.items():
        models, multis, recs, probs, metrics = rate_modulation(
            builder, edges, agonists, mul_range)

        # TODO: fix recs and probs before they are returned from rate_modulation
        # then adjust the plotting functions to expect this new data shape.
        def stack_recs(rs):
            return {
                k: np.stack([r[k] for r in rs], axis=0)
                for k in rs[0].keys()
            }

        data[label] = {
            "recs": {k: stack_recs(v) for k,v in recs.items()},
            "probs": {k: np.stack(v, axis=0) for k, v in probs.items()},
            "metrics": metrics
        }

        # edge_str = ", ".join([" → ".join(e) for e in edges])
        # title = "%s; %s; (Space %s)" % (models["prox"].name, edge_str, d_flag)
        title = "%s %s modulation" % (models["prox"].name, label)
        metric_fig = plot_modulation_metrics(multis, metrics, title=title)
        open_fig = plot_modulation_open_prob(
            multis, models["prox"].time, probs, title=title)
        states_fig = plot_modulation_states(
            multis, models["prox"].time, recs, title=title)
        fname = models["prox"].name + "_mod_" + label + "_" + d_flag
        metric_fig.savefig(pth + fname + "_metrics." + fmt, bbox_inches="tight")
        open_fig.savefig(pth + fname + "_open_prob." + fmt, bbox_inches="tight")
        states_fig.savefig(pth + fname + "_states." + fmt, bbox_inches="tight")

        if show:
            plt.show()

    data["time"] = np.array(models["prox"].time)
    data["multis"] = np.array(multis)

    model_name = ("%s" % models["prox"].name).replace(" ", "_")
    utils.pack_hdf(pth + model_name + "_rate_mod", data)


def kd_vs_peak_ratio():
    """Peak ratios obtained by dividing open peak open probabilities of
    receptors exposed to either glutamate or ACH release events at distal and
    proximal 2D cleft diffusion distances. Plotted against KD dissociation
    constants (koff / kon).
    """
    receptors = ["GABA", "alpha7", "alpha6ish", "alpha3", "NMDA"]
    kds = np.array([15, 2, 2, 36.5, 3.05])
    glut_rs = np.array([.268, .462, .608, .1, .774])
    ach_rs = np.array([.506, .735, .857, .199, .981])

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
    ax[0].scatter(kds, glut_rs)
    ax[1].scatter(kds, ach_rs)
    for i, r in enumerate(receptors):
        ax[0].annotate(r, (kds[i], glut_rs[i]))
        ax[1].annotate(r, (kds[i], ach_rs[i]))

    ax[0].set_ylabel("Distal / Prox Ratio")
    for a in ax:
        a.set_ylim(0, 1)
        a.set_xlabel("KD (μM)")
        a.spines["right"].set_visible(False)
        a.spines["top"].set_visible(False)
        for ticks in (a.get_yticklabels()):
            ticks.set_fontsize(11)

    ax[0].set_title("Glutamate Release", fontsize=14)
    ax[1].set_title("ACH Release", fontsize=14)

    fig.tight_layout()
    plt.show()


def plot_diffusion(trans="ach", radii=[0., 1.1], spaces=[2, 3], save_pth=None,
                   fmt="png"):
    time_ax = np.arange(1, 25001) * .001  # [ms]
    time = time_ax / 1000  # [s]

    funcs = {}
    if 2 in spaces:
        if trans == "ach":
            funcs["2D"] = {"%.1f" % r: ach_2D(r * 1e-6) for r in radii}
        else:
            funcs["2D"] = {"%.1f" % r: glut_2D(r * 1e-6) for r in radii}

    if 3 in spaces:
        if trans == "ach":
            funcs["3D"] = {"%.1f" % r: ach_3D(r * 1e-6) for r in radii}
        else:
            funcs["3D"] = {"%.1f" % r: glut_3D(r * 1e-6) for r in radii}

    fig, ax = plt.subplots(1, figsize=(5, 5))
    for space, rad_dict in funcs.items():
        for rad, fun in rad_dict.items():
            wave = fun(time) / 1000
            ax.plot(time_ax, wave, label="r = %sμm (%s)" % (rad, space))

    ax.set_yscale("log")
    ax.set_ylim(5e-7, 1e-3)
    ax.set_xlim(0, 5)
    ax.set_ylabel("Concentration (M)", fontsize=12)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.legend(frameon=False, fontsize=11)

    for ticks in (ax.get_xticklabels() + ax.get_yticklabels()):
        ticks.set_fontsize(11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    if save_pth is not None:
        fname = "diffusion_profiles_%s.%s" % (trans, fmt)
        fig.savefig(save_pth + fname, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    base_pth = "/mnt/Data/kinetics/"
    fig_pth = base_pth + "ach_2d/pdfs/"
    # fig_pth = base_pth + "new_diffusion/"
    if not os.path.isdir(fig_pth):
        os.mkdir(fig_pth)

    fig_fmt="pdf"

    diffusion2D_alpha_comparison(save_pth=fig_pth, fmt=fig_fmt)
    alpha7_vs_alpha6(save_pth=fig_pth, fmt=fig_fmt)

    # prox_vs_distal(gb.loader(gb.GABA))
    # prox_vs_distal(gb.loader(gb.AChSnfr))
    # prox_vs_distal(gb.loader(gb.alpha7), save_pth=fig_pth, title="Alpha 7 nACHR", fmt=fig_fmt)
    # prox_vs_distal(gb.loader(gb.alpha7, desens_div=4), threeD=False, save_pth=fig_pth,
    #                title="Alpha 6 nACHR", fmt=fig_fmt)
    # prox_vs_distal(gb.loader(gb.Pesti_alpha7))
    # prox_vs_distal(gb.loader(gb.McCormack_alpha7))
    # prox_vs_distal(gb.loader(gb.Mike_Circular_alpha7))
    # prox_vs_distal(gb.loader(gb.Mike_Bifurcated_alpha7))
    # prox_vs_distal(gb.loader(gb.alpha3), save_pth=fig_pth, title="Alpha 3", fmt=fig_fmt)
    # prox_vs_distal(gb.loader(gb.AMPAR), save_pth=fig_pth, title="AMPAR", fmt=fig_fmt)
    # prox_vs_distal(gb.loader(gb.Hatton_ACHR))
    # prox_vs_distal(gb.loader(gb.NMDAR, mode="M", tstop=200), save_pth=fig_pth,
    #                title="NMDAR", fmt=fig_fmt)

    # pulse_plot(gb.loader(gb.GABA), [(0, 2, 10)], "GABA; pulse: 2ms @ 10mM")
    # pulse_plot(gb.loader(gb.alpha7), [(0, 1, .01)], "a7; pulse: 1ms @ 10μM")
    # pulse_plot(gb.loader(gb.alpha7, desens_div=4),
    #            [(0, .125, .1)], "a6; pulse: 0.125ms @ 100μM")
    # pulse_plot(gb.loader(gb.alpha7, desens_div=4),
    #            [(0, .25, .1)], "a6; pulse: 0.25ms @ 100μM")
    # pulse_plot(gb.loader(gb.alpha7, desens_div=4),
    #            [(0, .5, .1)], "a6; pulse: 0.5ms @ 100μM")
    # pulse_plot(gb.loader(gb.alpha7, desens_div=4),
    #            [(0, 1, .1)], "a6; pulse: 1ms @ 100μM")
    # pulse_plot(gb.loader(gb.alpha7, desens_div=4),
    #            [(0, 2, .1)], "a6; pulse: 2ms @ 100μM")
    # pulse_plot(gb.loader(gb.alpha7, desens_div=4),
    #            [(0, 3, .1)], "a6; pulse: 3ms @ 100μM")
    # pulse_plot(gb.loader(gb.NMDAR, tstop=1000),
    #            [(0, .125, .1)], "NMDAR; pulse: 0.125ms @ 100μM")
    # pulse_plot(gb.loader(gb.NMDAR, tstop=1000),
    #            [(0, .25, .1)], "NMDAR; pulse: 0.25ms @ 100μM")
    # pulse_plot(gb.loader(gb.NMDAR, tstop=1000),
    #            [(0, .5, .1)], "NMDAR; pulse: 0.5ms @ 100μM")
    # pulse_plot(gb.loader(gb.NMDAR, tstop=1000),
    #            [(0, 1, .1)], "NMDAR; pulse: 1ms @ 100μM")
    # pulse_plot(gb.loader(gb.NMDAR, tstop=1000),
    #            [(0, 2, .1)], "NMDAR; pulse: 2ms @ 100μM")
    # pulse_plot(gb.loader(gb.NMDAR, tstop=1000),
    #            [(0, 3, .1)], "NMDAR; pulse: 3ms @ 100μM")
    # pulse_plot(gb.loader(gb.NMDAR, tstop=1000),
    #            [(0, 15, .1)], "NMDAR; pulse: 15ms @ 100μM")
    # plt.show()

    # binding_modulation_run(
    #     fig_pth, gb.loader(gb.AMPAR),
    #     mul_range=10, trans="ach", threeD=False
    # )

    loaders = [
        gb.loader(gb.alpha3),
        gb.loader(gb.alpha7),
        gb.loader(gb.alpha7, desens_div=4),
    ]
    # for ldr in loaders:
    #     binding_modulation_run(
    #         fig_pth, ldr, mul_range=10, trans="ach", threeD=False, fmt=fig_fmt
    #     )

    # prox_vs_distal_states(gb.loader(gb.alpha7))
    # prox_vs_distal_states(gb.loader(gb.alpha7, desens_div=4))
    # prox_vs_distal_states(gb.loader(gb.alpha3))
    # kd_vs_peak_ratio()

    # plot_diffusion(spaces=[2], save_pth=fig_pth, fmt="png")
