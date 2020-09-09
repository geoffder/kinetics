import numpy as np

from kinetic_graph import KineticGraph


def GABA(mods):
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="GABA")
    """
    Petrini et al., 2003
    Rate Constants:
        k1 = k2 = 8       [1 / (mM * ms)]
        k-1 = k-2 = 0.12  [1 / ms]
        beta1 = 0.04      [1 / ms]
        alpha1 = 0.2      [1 / ms]
        d1 = 0.013        [1 / ms]
        r1 = 0.0013       [1 / ms]
        beta2 = 3.45      [1 / ms]
        alpha2 = 1        [1 / ms]
        d2 = 1.45         [1 / ms]
        r2 = 0.1          [1 / ms]
    """
    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("AR*")       # single agonist bound open
    graph.add_node("AD")        # single agonist bound deactivated
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A2R*")      # double agonist bound open
    graph.add_node("A2D")       # double agonist bound deactivated

    graph.new_edge("R", "AR", 8., agonist_sens=2)    # kon * 2 (agonist bind)
    graph.new_edge("AR", "R", 0.12)                  # koff * 1 (agonist unbind)
    graph.new_edge("AR", "AD", 0.013)                # d1 (deactivate)
    graph.new_edge("AD", "AR", 0.0013)               # r1 (reactivate)
    graph.new_edge("AR", "AR*", 0.04)                # beta1 (channel open)
    graph.new_edge("AR*", "AR", 0.2)                 # alpha1 (channel close)
    graph.new_edge("AR", "A2R", 8., agonist_sens=1)  # kon * 1 (agonist bind)
    graph.new_edge("A2R", "AR", 0.12 * 2)            # koff * 2 (agonist unbind)
    graph.new_edge("A2R", "A2D", 1.45)               # d2 (deactivate)
    graph.new_edge("A2D", "A2R", 0.1)                # r2 (reactivate)
    graph.new_edge("A2R", "A2R*", 3.45)              # beta2 (channel open)
    graph.new_edge("A2R*", "A2R", 1.)                # alpha2 (channel close)

    return graph


def AMPAR(mods):
    """
    Haas et al., 2018
    https://elifesciences.org/articles/31755#fig6
    Note that kon and koff rates were fit independently at each edge
    (e.g. they were free parameters) rather than being fit once each and applied
    to all relevant edges, scaled according to the number of agonist slots
    available for binding/unbinding.
    Rate Constants:
       k1 = 13.77      [1 / (μM * s)]
       k1r = 2130      [1 / s]
       k2 = 85.2       [1 / (μM * s)]
       k2r = 1630      [1 / s]
       k3 = 3.81       [1 / (μM * s)]
       k3r = 22.85     [1 / s]
       alpha = 1696    [1 / s]
       beta = 630      [1 / s]
       alpha1 = 1445   [1 / s]
       beta1 = 19.6    [1 / s]
       alpha2 = 86     [1 / s]
       beta2 = 0.3635  [1 / s]
       alpha3 = 8.85   [1 / s]
       beta3 = 2       [1 / s]
       alpha4 = 6.72   [1 / s]
       beta4 = 133.28  [1 / s]
    [1 / (μM * s)] rates equivalent to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="AMPAR")

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A2R*")      # double agonist bound open
    graph.add_node("AD")        # single agonist bound desensitized
    graph.add_node("A2D")       # double agonist bound desensitized
    graph.add_node("A2Dp")      # double agonist bound desensitized prime

    graph.new_edge("R", "AR", 13.77, 1)     # k1 (agonist bind)
    graph.new_edge("AR", "R", 2.13)         # k1r (agonist unbind)
    graph.new_edge("AR", "A2R", 85.2, 1)    # k2 (agonist bind)
    graph.new_edge("A2R", "AR", 1.63)       # k2r (agonist unbind)
    graph.new_edge("A2R", "A2R*", 1.696)    # alpha (opening)
    graph.new_edge("A2R*", "A2R", .63)      # beta (closing)

    graph.new_edge("AR", "AD", 1.445)       # alpha1 (desensitize)
    graph.new_edge("AD", "AR", .0196)       # beta1 (resensitize)
    graph.new_edge("A2R", "A2D", .086)      # alpha2 (desensitize)
    graph.new_edge("A2D", "A2R", .0003635)  # beta2 (resensitize)
    graph.new_edge("A2R*", "A2Dp", .00885)  # alpha3  (desensitize)
    graph.new_edge("A2Dp", "A2R*", .002)    # beta3 (resensitize)

    graph.new_edge("AD", "A2D", 3.81, 1)    # k3 (agonist bind)
    graph.new_edge("A2D", "AD", .02285)     # k3r (agonist unbind)
    graph.new_edge("A2D", "A2Dp", .00672)   # alpha4 (prime)
    graph.new_edge("A2Dp", "A2D", .13328)   # beta4 (de-prime)

    return graph


def NMDAR(mods):
    """
    Popescu et al., 2004. doi: 10.1038/nature02775.
    Reaction mechanism determines NMDA receptor response to repetitive stim..
    Modal rate constants (M is most common mode):
               H     M     L
        kon  = 20    19    17     [1 / (μM * s)]
        koff = 58    58    60     [1 / s]
        C1C2 = 93    150   127    [1 / s]
        C2C1 = 196   173   161    [1 / s]
        C2C3 = 914   902   580    [1 / s]
        C3C2 = 954   2412  2610   [1 / s]
        C3O1 = 6729  4467  2508   [1 / s]
        O1C3 = 321   1283  2167   [1 / s]
        O1O2 = 1343  4630  3449   [1 / s]
        O2O1 = 247   526   662    [1 / s]
    [1 / (μM * s)] rates equivalent to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    mode = mods.get("mode", "M")
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="NMDAR_%s" % mode)

    if mode == "M":
        kon = 19
        koff = .058
        C1C2 = .150
        C2C1 = .173
        C2C3 = .902
        C3C2 = 2.412
        C3O1 = 4.467
        O1C3 = 1.283
        O1O2 = 4.630
        O2O1 = .526
    elif mode == "H":
        kon = 20
        koff = .058
        C1C2 = .093
        C2C1 = .196
        C2C3 = .914
        C3C2 = .954
        C3O1 = 6.729
        O1C3 = .321
        O1O2 = 1.343
        O2O1 = .247
    else:
        kon = 17
        koff = .060
        C1C2 = .127
        C2C1 = .161
        C2C3 = .580
        C3C2 = 2.610
        C3O1 = 2.508
        O1C3 = 2.167
        O1O2 = 3.449
        O2O1 = .662

    graph.add_node("R", v0=1.)  # unbound ready (Cu, unliganded)
    graph.add_node("AR")        # singly bound ready (Cm, monoliganded)
    graph.add_node("A2R1")      # doubly bound closed 1 (C1)
    graph.add_node("A2R2")      # doubly bound closed 2 (C2)
    graph.add_node("A2R3")      # doubly bound closed 3 (C3)
    graph.add_node("A2R*")      # doubly bound open 1 (O1)
    graph.add_node("A2R**")     # doubly bound open 2 (O2)

    graph.new_edge("R", "AR", kon, agonist_sens=2)     # 2kon (agonist bind)
    graph.new_edge("AR", "R", koff)                    # koff (agonist unbind)
    graph.new_edge("AR", "A2R1", kon, agonist_sens=1)  # kon (agonist bind)
    graph.new_edge("A2R1", "AR", koff * 2)             # 2koff (agonist unbind)

    graph.new_edge("A2R1", "A2R2", C1C2)               # C1C2 (prime)
    graph.new_edge("A2R2", "A2R1", C2C1)               # C2C1 (de-prime)
    graph.new_edge("A2R2", "A2R3", C2C3)               # C2C3 (prime)
    graph.new_edge("A2R3", "A2R2", C3C2)               # C3C2 (de-prime)
    graph.new_edge("A2R3", "A2R*", C3O1)               # C3O1 (open)
    graph.new_edge("A2R*", "A2R3", O1C3)               # O1C3 (close)
    graph.new_edge("A2R*", "A2R**", O1O2)              # O1O2 (lodge open)
    graph.new_edge("A2R**", "A2R*", O2O1)              # O2O1 (de-lodge)

    return graph


def muscle_ACHR(mods):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1260158/pdf/biophysj00108-0022.pdf
    Franke et al., 1991
    Parameters fit for reaction #1:
       k1 = k2 = 1e8     [1 / (M * s)]
       k-1 = k-2 = 20000 [1 / s]
       alpha = 1100      [1 / s]
       beta = 50000      [1 / s]
    kon rates adjusted by 1e6 to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="Muscle ACHR")

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A2R*")      # double agonist bound open

    graph.new_edge("R", "AR", 100., agonist_sens=2)    # k+1 (agonist bind)
    graph.new_edge("AR", "R", 20.)                     # k-1 (agonist unbind)
    graph.new_edge("AR", "A2R", 100., agonist_sens=1)  # k+2 (agonist bind)
    graph.new_edge("A2R", "AR", 20. * 2)               # k-2 (agonist unbind)
    graph.new_edge("A2R", "A2R*", 50.)                 # beta (opening)
    graph.new_edge("A2R*", "A2R", 1.1)                 # alpha (closing)

    return graph


def Hatton_ACHR(mods):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2342726/
    Hatton et al., 2003
    Wild-type.
    Parameters fit (EC50 constrained) for scheme 1:
        alpha2 = 2460         [1 / s]
        beta2 = 51600         [1 / s]
        alpha1a = 6000        [1 / s]
        beta1a = 37.7         [1 / s]
        alpha1b = 71300       [1 / s]
        beta1b = 156          [1 / s]
        k-1a = k-2a = 1330    [1 / s]
        k+1a = k+2a = 2.23e7  [1 / (M * s)]
        k-1b = k-2b = 13400   [1 / s]
        k+1b = k+2b = 4.7e8   [1 / (M * s)]
    [1 / (M * s)] rates adjusted by 1e6 to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="Hatton ACHR")

    graph.add_node("Ra-Rb", v0=1.)  # unbound ready
    graph.add_node("ARa-Rb")        # Ra agonist bound ready
    graph.add_node("Ra-ARb")        # Rb agonist bound ready
    graph.add_node("ARa-Rb*")       # Ra agonist bound open
    graph.add_node("Ra-ARb*")       # Rb agonist bound open
    graph.add_node("A2R")           # double agonist bound ready
    graph.add_node("A2R*")          # double agonist bound open
    graph.add_node("A2D")           # double agonist bound desensitized

    graph.new_edge("Ra-Rb", "ARa-Rb", 22.3, 1)  # k+1a (agonist bind)
    graph.new_edge("ARa-Rb", "Ra-Rb", 1.33)     # k-1a (Ra agonist unbind)
    graph.new_edge("ARa-Rb", "ARa-Rb*", .0377)  # beta1a (ARa open)
    graph.new_edge("ARa-Rb*", "ARa-Rb", 6.)     # alpha1a (ARa close)
    graph.new_edge("ARa-Rb", "A2R", 470., 1)    # k+2b (Rb agonist bind)
    graph.new_edge("A2R", "ARa-Rb", 13.4)       # k-2b (Rb agonist unbind)

    graph.new_edge("Ra-Rb", "Ra-ARb", 470., 1)  # k+1b (agonist bind)
    graph.new_edge("Ra-ARb", "Ra-Rb", 13.4)     # k-1b (Rb agonist unbind)
    graph.new_edge("Ra-ARb", "Ra-ARb*", .156)   # beta1b (ARb open)
    graph.new_edge("Ra-ARb*", "Ra-ARb", 71.3)   # alpha1b (ARb close)
    graph.new_edge("Ra-ARb", "A2R", 22.3, 1)    # k+2a (Ra agonist bind)
    graph.new_edge("A2R", "Ra-ARb", 1.33)       # k-2a (Ra agonist unbind)

    graph.new_edge("A2R", "A2R*", 51.6)         # beta2 (double bound open)
    graph.new_edge("A2R*", "A2R", 2.46)         # alpha2 (double bound close)

    return graph


def Hatton_L221F_ACHR(mods):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2342726/
    Hatton et al., 2003
    L221F Mutant.
    Parameters fit (EC50 constrained) for scheme2:
        alpha2 = 1440         [1 / s]
        beta2 = 85000         [1 / s]
        alpha1a = 3110        [1 / s]
        beta1a = 12.2         [1 / s]
        alpha1b = 57000       [1 / s]
        beta1b = 14.6         [1 / s]
        k-1a = k-2a = 409     [1 / s]
        k+1a = k+2a = 1.74e7  [1 / (M * s)]
        k-1b = k-2b = 3210    [1 / s]
        k+1b = k+2b = 3.69e8  [1 / (M * s)]
        des = 19.3            [1 / s]
        res = 9400            [1 / s]
    [1 / (M * s)] rates adjusted by 1e6 to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="Hatton ACHR L221F")

    graph.add_node("Ra-Rb", v0=1.)  # unbound ready
    graph.add_node("ARa-Rb")        # Ra agonist bound ready
    graph.add_node("Ra-ARb")        # Rb agonist bound ready
    graph.add_node("ARa-Rb*")       # Ra agonist bound open
    graph.add_node("Ra-ARb*")       # Rb agonist bound open
    graph.add_node("A2R")           # double agonist bound ready
    graph.add_node("A2R*")          # double agonist bound open
    graph.add_node("A2D")           # double agonist bound desensitized

    graph.new_edge("Ra-Rb", "ARa-Rb", 17.4, 1)  # k+1a (agonist bind)
    graph.new_edge("ARa-Rb", "Ra-Rb", .409)     # k-1a (Ra agonist unbind)
    graph.new_edge("ARa-Rb", "ARa-Rb*", .0122)  # beta1a (ARa open)
    graph.new_edge("ARa-Rb*", "ARa-Rb", 3.11)   # alpha1a (ARa close)
    graph.new_edge("ARa-Rb", "A2R", 369., 1)    # k+2b (Rb agonist bind)
    graph.new_edge("A2R", "ARa-Rb", 3.21)       # k-2b (Rb agonist unbind)

    graph.new_edge("Ra-Rb", "Ra-ARb", 369., 1)  # k+1b (agonist bind)
    graph.new_edge("Ra-ARb", "Ra-Rb", 3.21)     # k-1b (Rb agonist unbind)
    graph.new_edge("Ra-ARb", "Ra-ARb*", .0146)  # beta1b (ARb open)
    graph.new_edge("Ra-ARb*", "Ra-ARb", 57.)    # alpha1b (ARb close)
    graph.new_edge("Ra-ARb", "A2R", 17.4, 1)    # k+2a (Ra agonist bind)
    graph.new_edge("A2R", "Ra-ARb", .409)       # k-2a (Ra agonist unbind)

    graph.new_edge("A2R", "A2R*", 85.)          # beta2 (double bound open)
    graph.new_edge("A2R*", "A2R", 1.44)         # alpha2 (double bound close)
    graph.new_edge("A2R*", "A2D", .0193)        # des (desensitize)
    graph.new_edge("A2D", "A2R*", 9.4)          # res (resensitize)

    return graph


def Pesti_alpha7(mods):
    """
    Pesti et al., 2014
    Rate Constants:
       X = 5                               [constant]
       Y = 2                               [constant]
       Z = 20                              [constant]
       kon (ax) = 10 * (5 - x)             [1 / (μM * s)]
       koff (drx) = 500 * x                [1 / s]
       D-koff (ddx) = drx / X^2            [1 / s]
       S-koff (dsx) = ddx / Y^2            [1 / s]
       beta (ox) = 0.05 * Z^(x - 1)        [1 / s]
       alpha (cx) = 20000                  [1 / s]
       desens (dx) = 5 * X^(x - 1)         [1 / s]
       resens (rx) = 5000 / X^(x - 1)      [1 / s]
       slow-desens (sx) = .02 * Y^(x - 1)  [1 / s]
       slow-resens (rsx) = 1 / Y^(x - 1)   [1 / s]
    x = number of agonist bound
    [1 / (μM * s)] rates equivalent to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    flag = "_desensDiv%i" % mods["desens_div"] if "desens_div" in mods else ""
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="Pesti_alpha7" + flag)

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A3R")       # triple agonist bound ready
    graph.add_node("A4R")       # quadruple agonist bound ready
    graph.add_node("A5R")       # quintuple agonist bound ready
    graph.add_node("AR*")       # single agonist bound open
    graph.add_node("A2R*")      # double agonist bound open
    graph.add_node("A3R*")      # triple agonist bound open
    graph.add_node("A4R*")      # quadruple agonist bound open
    graph.add_node("A5R*")      # quintuple agonist bound open
    graph.add_node("D")         # unbound desensitized
    graph.add_node("AD")        # single agonist bound desensitized
    graph.add_node("A2D")       # double agonist bound desensitized
    graph.add_node("A3D")       # triple agonist bound desensitized
    graph.add_node("A4D")       # quadruple agonist bound desensitized
    graph.add_node("A5D")       # quintuple agonist bound desensitized
    graph.add_node("S")         # unbound slow desensitized
    graph.add_node("AS")        # single agonist bound slow desensitized
    graph.add_node("A2S")       # double agonist bound slow desensitized
    graph.add_node("A3S")       # triple agonist bound slow desensitized
    graph.add_node("A4S")       # quadruple agonist bound slow desensitized
    graph.add_node("A5S")       # quintuple agonist bound slow desensitized

    X = 5.
    Y = 2.
    Z = 20.

    kon = 10. * mods.get("on_multi", 1)
    koff = .5
    beta = .00005
    alpha = 20.
    des = .005 / mods.get("desens_div", 1)
    res = 5.
    slow_des = .00002
    slow_res = .001

    graph.new_edge("R", "AR", kon, 5)              # a1 (bind)
    graph.new_edge("AR", "R", koff)                # dr1 (unbind)
    graph.new_edge("AR", "A2R", kon, 4)            # a2 (bind)
    graph.new_edge("A2R", "AR", koff * 2)          # dr2 (unbind)
    graph.new_edge("A2R", "A3R", kon, 3)           # a3 (bind)
    graph.new_edge("A3R", "A2R", koff * 3)         # dr3 (unbind)
    graph.new_edge("A3R", "A4R", kon, 2)           # a4 (bind)
    graph.new_edge("A4R", "A3R", koff * 4)         # dr4 (unbind)
    graph.new_edge("A4R", "A5R", kon, 1)           # a5 (bind)
    graph.new_edge("A5R", "A4R", koff * 5)         # dr5 (unbind)

    D_koff = koff / X**2
    graph.new_edge("D", "AD", kon, 5)              # a1 (bind)
    graph.new_edge("AD", "D", D_koff)              # dd1 (unbind)
    graph.new_edge("AD", "A2D", kon, 4)            # a2 (bind)
    graph.new_edge("A2D", "AD", D_koff * 2)        # dd2 (unbind)
    graph.new_edge("A2D", "A3D", kon, 3)           # a3 (bind)
    graph.new_edge("A3D", "A2D", D_koff * 3)       # dd3 (unbind)
    graph.new_edge("A3D", "A4D", kon, 2)           # a4 (bind)
    graph.new_edge("A4D", "A3D", D_koff * 4)       # dd4 (unbind)
    graph.new_edge("A4D", "A5D", kon, 1)           # a5 (bind)
    graph.new_edge("A5D", "A4D", D_koff * 5)       # dd5 (unbind)

    S_koff = D_koff / Y**2
    graph.new_edge("S", "AS", kon, 5)              # a1 (bind)
    graph.new_edge("AS", "S", S_koff)              # ds1 (unbind)
    graph.new_edge("AS", "A2S", kon, 4)            # a2 (bind)
    graph.new_edge("A2S", "AS", S_koff * 2)        # ds2 (unbind)
    graph.new_edge("A2S", "A3S", kon, 3)           # a3 (bind)
    graph.new_edge("A3S", "A2S", S_koff * 3)       # ds3 (unbind)
    graph.new_edge("A3S", "A4S", kon, 2)           # a4 (bind)
    graph.new_edge("A4S", "A3S", S_koff * 4)       # ds4 (unbind)
    graph.new_edge("A4S", "A5S", kon, 1)           # a5 (bind)
    graph.new_edge("A5S", "A4S", S_koff * 5)       # ds5 (unbind)

    graph.new_edge("R", "D", des)                  # d0 (desensitize)
    graph.new_edge("D", "R", res)                  # r0 (resensitize)
    graph.new_edge("D", "S", slow_des)             # s0 (slow desensitize)
    graph.new_edge("S", "D", slow_res)             # rs0 (slow resensitize)

    graph.new_edge("AR", "AR*", beta)              # o1 (open)
    graph.new_edge("AR*", "AR", alpha)             # c1 (close)
    graph.new_edge("AR", "AD", des * X)            # d1 (desensitize)
    graph.new_edge("AD", "AR", res / X)            # r1 (resensitize)
    graph.new_edge("AD", "AS", slow_des * Y)       # s1 (slow desensitize)
    graph.new_edge("AS", "AD", slow_res / Y)       # rs1 (slow resensitize)

    graph.new_edge("A2R", "A2R*", beta * Z)        # o2 (open)
    graph.new_edge("A2R*", "A2R", alpha)           # c2 (close)
    graph.new_edge("A2R", "A2D", des * X**2)       # d2 (desensitize)
    graph.new_edge("A2D", "A2R", res / X**2)       # r2 (resensitize)
    graph.new_edge("A2D", "A2S", slow_des * Y**2)  # s2 (slow desensitize)
    graph.new_edge("A2S", "A2D", slow_res / Y**2)  # rs2 (slow resensitize)

    graph.new_edge("A3R", "A3R*", beta * Z**2)     # o3 (open)
    graph.new_edge("A3R*", "A3R", alpha)           # c3 (close)
    graph.new_edge("A3R", "A3D", des * X**3)       # d3 (desensitize)
    graph.new_edge("A3D", "A3R", res / X**3)       # r3 (resensitize)
    graph.new_edge("A3D", "A3S", slow_des * Y**3)  # s3 (slow desensitize)
    graph.new_edge("A3S", "A3D", slow_res / Y**3)  # rs3 (slow resensitize)

    graph.new_edge("A4R", "A4R*", beta * Z**3)     # o4 (open)
    graph.new_edge("A4R*", "A4R", alpha)           # c4 (close)
    graph.new_edge("A4R", "A4D", des * X**4)       # d4 (desensitize)
    graph.new_edge("A4D", "A4R", res / X**4)       # r4 (resensitize)
    graph.new_edge("A4D", "A4S", slow_des * Y**4)  # s4 (slow desensitize)
    graph.new_edge("A4S", "A4D", slow_res / Y**4)  # rs4 (slow resensitize)

    graph.new_edge("A5R", "A5R*", beta * Z**4)     # o5 (open)
    graph.new_edge("A5R*", "A5R", alpha)           # c5 (close)
    graph.new_edge("A5R", "A5D", des * X**5)       # d5 (desensitize)
    graph.new_edge("A5D", "A5R", res / X**5)       # r5 (resensitize)
    graph.new_edge("A5D", "A5S", slow_des * Y**5)  # s5 (slow desensitize)
    graph.new_edge("A5S", "A5D", slow_res / Y**5)  # rs5 (slow resensitize)

    return graph


def McCormack_alpha7(mods):
    """
    McCormack et al., 2010
    Rate Constants:
       kon (arx, adx) = 80 * (3 - x)  [1 / (μM * s)]
       koff (drx) = 10000 * x         [1 / s]
       D_koff (ddx) = 2.778 * x       [1 / s]
       beta (ox) = 50                 [1 / s]
       alpha (cx) = 2500              [1 / s]
       desens
         d0 = .0002                   [1 / s]
         d1 = .012                    [1 / s]
         d2 = 10                      [1 / s]
         d3 = 10                      [1 / s]
       resens
         r0 = 1                       [1 / s]
         r1 = .01667                  [1 / s]
         r2 = .00386                  [1 / s]
         r3 = 1.1e-6                  [1 / s]
    x = number of agonist bound
    [1 / (μM * s)] rates equivalent to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    flag = "_desensDiv%i" % mods["desens_div"] if "desens_div" in mods else ""
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="McCormack_alpha7" + flag)

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A3R")       # triple agonist bound ready
    graph.add_node("A2R*")      # double agonist bound open
    graph.add_node("A3R*")      # triple agonist bound open
    graph.add_node("D")         # unbound desensitized
    graph.add_node("AD")        # single agonist bound desensitized
    graph.add_node("A2D")       # double agonist bound desensitized
    graph.add_node("A3D")       # triple agonist bound desensitized

    kon = 80. * mods.get("on_multi", 1)
    koff = 10.
    D_koff = .002778
    beta = .05
    alpha = 2.5
    des = np.array([2e-7, 12e-6, .01, .01]) / mods.get("desens_div", 1)
    res = np.array([.001, 16.67e-6, 3.86e-6, 1.1e-9])

    graph.new_edge("R", "AR", kon, 3)         # ar1 (bind)
    graph.new_edge("AR", "R", koff)           # dr1 (unbind)
    graph.new_edge("AR", "A2R", kon, 2)       # ar2 (bind)
    graph.new_edge("A2R", "AR", koff * 2)     # dr2 (unbind)
    graph.new_edge("A2R", "A3R", kon, 1)      # ar3 (bind)
    graph.new_edge("A3R", "A2R", koff * 3)    # dr3 (unbind)

    graph.new_edge("D", "AD", kon, 3)         # ad1 (bind)
    graph.new_edge("AD", "D", D_koff)         # dd1 (unbind)
    graph.new_edge("AD", "A2D", kon, 2)       # ad2 (bind)
    graph.new_edge("A2D", "AD", D_koff * 2)   # dd2 (unbind)
    graph.new_edge("A2D", "A3D", kon, 1)      # ad3 (bind)
    graph.new_edge("A3D", "A2D", D_koff * 3)  # dd3 (unbind)

    graph.new_edge("R", "D", des[0])          # d0 (desensitize)
    graph.new_edge("D", "R", res[0])          # r0 (resensitize)
    graph.new_edge("AR", "AD", des[1])        # d1 (desensitize)
    graph.new_edge("AD", "AR", res[1])        # r1 (resensitize)
    graph.new_edge("A2R", "A2D", des[2])      # d2 (desensitize)
    graph.new_edge("A2D", "A2R", res[2])      # r2 (resensitize)
    graph.new_edge("A3R", "A3D", des[3])      # d3 (desensitize)
    graph.new_edge("A3D", "A3R", res[3])      # r3 (resensitize)

    graph.new_edge("A2R", "A2R*", beta)       # o2 (open)
    graph.new_edge("A2R*", "A2R", alpha)      # c2 (close)
    graph.new_edge("A3R", "A3R*", beta)       # o3 (open)
    graph.new_edge("A3R*", "A3R", alpha)      # c3 (close)

    return graph


def Mike_Circular_alpha7(mods):
    """
    Mike et al., 2000
    Circular model (Fig 8A, G configuration)
    Rate Constants:
       kon (arx, adx) = 100 * (3 - x)  [1 / (μM * s)]
       koff (drx) = 10000 * x          [1 / s]
       D_koff (ddx) = 20 * x           [1 / s]
       beta (ox) = 10000               [1 / s]
       alpha (cx) = 8000               [1 / s]
       desens
         d0 = .01                      [1 / s]
         d2 = 5000                     [1 / s]
         d3 = 5000                     [1 / s]
       resens
         r0 = 500                      [1 / s]
         r2 = 50                       [1 / s]
         r3 = 1                        [1 / s]
    x = number of agonist bound
    [1 / (μM * s)] rates equivalent to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    flag = "_desensDiv%i" % mods["desens_div"] if "desens_div" in mods else ""
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="Mike_Circular_alpha7" + flag)

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A3R")       # triple agonist bound ready
    graph.add_node("A2R*")      # double agonist bound open
    graph.add_node("A3R*")      # triple agonist bound open
    graph.add_node("D")         # unbound desensitized
    graph.add_node("AD")        # single agonist bound desensitized
    graph.add_node("A2D")       # double agonist bound desensitized
    graph.add_node("A3D")       # triple agonist bound desensitized

    kon = 100. * mods.get("on_multi", 1)
    koff = 10.
    D_koff = .02
    beta = 10.
    alpha = 8.
    des = np.array([.00001, np.nan, 5., 5.]) / mods.get("desens_div", 1)
    res = np.array([.5, np.nan, .05, .001])

    graph.new_edge("R", "AR", kon, 3)         # ar1 (bind)
    graph.new_edge("AR", "R", koff)           # dr1 (unbind)
    graph.new_edge("AR", "A2R", kon, 2)       # ar2 (bind)
    graph.new_edge("A2R", "AR", koff * 2)     # dr2 (unbind)
    graph.new_edge("A2R", "A3R", kon, 1)      # ar3 (bind)
    graph.new_edge("A3R", "A2R", koff * 3)    # dr3 (unbind)

    graph.new_edge("D", "AD", kon, 3)         # ad1 (bind)
    graph.new_edge("AD", "D", D_koff)         # dd1 (unbind)
    graph.new_edge("AD", "A2D", kon, 2)       # ad2 (bind)
    graph.new_edge("A2D", "AD", D_koff * 2)   # dd2 (unbind)
    graph.new_edge("A2D", "A3D", kon, 1)      # ad3 (bind)
    graph.new_edge("A3D", "A2D", D_koff * 3)  # dd3 (unbind)

    graph.new_edge("R", "D", des[0])          # d0 (desensitize)
    graph.new_edge("D", "R", res[0])          # r0 (resensitize)
    graph.new_edge("A2R*", "A2D", des[2])     # d2 (desensitize)
    graph.new_edge("A2D", "A2R*", res[2])     # r2 (resensitize)
    graph.new_edge("A3R*", "A3D", des[3])     # d3 (desensitize)
    graph.new_edge("A3D", "A3R*", res[3])     # r3 (resensitize)

    graph.new_edge("A2R", "A2R*", beta)       # o2 (open)
    graph.new_edge("A2R*", "A2R", alpha)      # c2 (close)
    graph.new_edge("A3R", "A3R*", beta)       # o3 (open)
    graph.new_edge("A3R*", "A3R", alpha)      # c3 (close)

    return graph


def Mike_Bifurcated_alpha7(mods):
    """
    Mike et al., 2000
    Bifurcated model (Fig 8B, G configuration)
    Rate Constants:
       kon (arx, adx) = 100 * (3 - x)  [1 / (μM * s)]
       koff (drx) = 10000 * x          [1 / s]
       beta (ox) = 10000               [1 / s]
       alpha (cx) = 10000              [1 / s]
       desens
         d2 = 5000                     [1 / s]
         d3 = 5000                     [1 / s]
       resens
         r2 = 5                        [1 / s]
         r3 = 1                        [1 / s]
    x = number of agonist bound
    [1 / (μM * s)] rates equivalent to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    flag = "_desensDiv%i" % mods["desens_div"] if "desens_div" in mods else ""
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="Mike_Bifurcated_alpha7" + flag)

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A3R")       # triple agonist bound ready
    graph.add_node("A2R*")      # double agonist bound open
    graph.add_node("A3R*")      # triple agonist bound open
    graph.add_node("D")         # unbound desensitized
    graph.add_node("AD")        # single agonist bound desensitized
    graph.add_node("A2D")       # double agonist bound desensitized
    graph.add_node("A3D")       # triple agonist bound desensitized

    kon = 100. * mods.get("on_multi", 1)
    koff = 10.
    beta = 10.
    alpha = 10.
    des = np.array([np.nan, np.nan, 5., 5.]) / mods.get("desens_div", 1)
    res = np.array([np.nan, np.nan, .05, .001])

    graph.new_edge("R", "AR", kon, 3)       # ar1 (bind)
    graph.new_edge("AR", "R", koff)         # dr1 (unbind)
    graph.new_edge("AR", "A2R", kon, 2)     # ar2 (bind)
    graph.new_edge("A2R", "AR", koff * 2)   # dr2 (unbind)
    graph.new_edge("A2R", "A3R", kon, 1)    # ar3 (bind)
    graph.new_edge("A3R", "A2R", koff * 3)  # dr3 (unbind)

    graph.new_edge("A2R", "A2D", des[2])    # d2 (desensitize)
    graph.new_edge("A2D", "A2R", res[2])    # r2 (resensitize)
    graph.new_edge("A3R", "A3D", des[3])    # d3 (desensitize)
    graph.new_edge("A3D", "A3R", res[3])    # r3 (resensitize)

    graph.new_edge("A2R", "A2R*", beta)     # o2 (open)
    graph.new_edge("A2R*", "A2R", alpha)    # c2 (close)
    graph.new_edge("A3R", "A3R*", beta)     # o3 (open)
    graph.new_edge("A3R*", "A3R", alpha)    # c3 (close)

    return graph


def alpha7(mods):
    """
    Coggan et al., 2005
    Rate Constants:
       k1 = k2 = 4.1e7   [1 / (M * s)]
       k-1 = k-2 = 82.2  [1 / s]
       des = 879         [1 / s]
       res = 26          [1 / s]
       beta = 86.2       [1 / s]
       alpha = 7641      [1 / s]
    kon rates adjusted by 1e6 to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    name = "alpha7"
    if "desens_div" in mods and mods["desens_div"] == 4:
        flag = ""
        name = "alpha6"
    elif "desens_div" in mods:
        flag = "_desensDiv%i" % mods["desens_div"]
    else:
        flag = ""
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name=name + flag)

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A2D")       # double agonist bound desensitized
    graph.add_node("A2R*")      # double agonist bound open

    kon = 41. * mods.get("on_multi", 1)
    koff = .0822
    des = .879 / mods.get("desens_div", 1)

    graph.new_edge("R", "AR", kon, agonist_sens=2)    # k+1 (bind)
    graph.new_edge("AR", "R", koff)                   # k-1 (unbind)
    graph.new_edge("AR", "A2R", kon, agonist_sens=1)  # k+2 (bind)
    graph.new_edge("A2R", "AR", koff * 2)             # k-2 (unbind)
    graph.new_edge("A2R", "A2D", des)                 # des (desensitize)
    graph.new_edge("A2D", "A2R", .026)                # res (resensitize)
    graph.new_edge("A2R", "A2R*", .0862)              # beta (opening)
    graph.new_edge("A2R*", "A2R", 7.641)              # alpha (closing)

    return graph


def alpha3(mods):
    """
    Coggan et al., 2005
    Rate Constants:
       k1 = k2 = 2.3e6  [1 / (M * s)]
       k-1 = k-2 = 84   [1 / s]
       beta = 513       [1 / s]
       alpha = 1000     [1 / s]
    kon rates adjusted by 1e6 to [1 / (mM * ms)].
    [1 / s] rates adjusted by 1e3 to [1 / ms].
    """
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="alpha3")

    graph.add_node("R", v0=1.)  # unbound ready
    graph.add_node("AR")        # single agonist bound ready
    graph.add_node("A2R")       # double agonist bound ready
    graph.add_node("A2R*")      # double agonist bound open

    kon = 2.3
    koff = .084

    graph.new_edge("R", "AR", kon, agonist_sens=2)    # k+1 (agonist bind)
    graph.new_edge("AR", "R", koff)                   # k-1 (agonist unbind)
    graph.new_edge("AR", "A2R", kon, agonist_sens=1)  # k+2 (agonist bind)
    graph.new_edge("A2R", "AR", koff * 2)             # k-2 (agonist unbind)
    graph.new_edge("A2R", "A2R*", .513)               # beta (opening)
    graph.new_edge("A2R*", "A2R", 1.)                 # alpha (closing)

    return graph


def AChSnfr(mods):
    """
    Borden et al., 2020
    Rate Constants:
        kon = 0.62   [1 / (μM * s)]
        koff = 0.73  [1 / s]
    """
    graph = KineticGraph(dt=mods.get("dt", .001),
                         tstop=mods.get("tstop", 25),
                         name="AChSnfr")

    graph.add_node("R", v0=1.)  # unbound
    graph.add_node("AR*")       # agonist bound

    graph.new_edge("R", "AR*", .62, agonist_sens=1)  # kon (agonist bind)
    graph.new_edge("AR*", "R", 0.73e-3)              # koff (agonist unbind)

    return graph


def loader(builder, **mods):
    """Takes function that creates a kinetic graph (with dict of mods), and
    returns a function that will build the graph and set it's agonist."""
    def closure(agonist_func=None, pulses=[]):
        graph = builder(mods)

        if agonist_func is not None:
            graph.diffusion_agonist(agonist_func)
        else:
            graph.pulse_agonist(pulses)  # concentration over time

        return graph

    return closure
