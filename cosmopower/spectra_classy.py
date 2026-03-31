from .parser import YAMLParser
import numpy as np
from classy import Class


def initialize(parser, extra_args={}):

    cosmo = Class()

    state = {}

    for quantity in parser.quantities:
        state[quantity + ".modes"] = parser.modes(quantity)

    state["code"] = "classy"
    state["module"] = cosmo
    state["params"] = extra_args
    state["derived"] = {}

    return state


def get_spectra(parser, state, args={}, quantities=[], extra_args={}):

    cosmo = state["module"]

    params = state["params"].copy()
    params.update(args)

    if "ln10^{10}A_s" in params:

        params["A_s"] = 1.e-10 * np.exp(params["ln10^{10}A_s"])

        del params["ln10^{10}A_s"]

    cosmo.set(params)

    cosmo.compute()

    cls = cosmo.raw_cl()
    ell = cls["ell"]

    for quantity in quantities:

        spec = quantity.split("/")[1]

        Cl = cls[spec]

        Dl = Cl * (ell * (ell + 1) / (2 * np.pi))

        state[quantity] = Dl[state[quantity + ".modes"]]

    cosmo.struct_cleanup()
    cosmo.empty()

    return True
