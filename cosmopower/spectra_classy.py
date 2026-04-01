from .parser import YAMLParser
import numpy as np


def initialize(parser: YAMLParser, extra_args: dict = {}) -> dict:
    import classy

    state = {}

    for quantity in parser.quantities:
        state[quantity + ".modes"] = parser.modes(quantity)

    extra_args["l_max_scalars"] = extra_args.get("l_max_scalars", 2)
    for quantity in parser.quantities:
        if parser.modes_label(quantity) == "l" and \
           extra_args["l_max_scalars"] < parser.modes(quantity).max():
           extra_args["l_max_scalars"] = parser.modes(quantity).max()

    state["code"] = "classy"
    state["module"] = classy
    state["params"] = extra_args
    state["derived"] = {}

    return state


def get_spectra(parser: YAMLParser, state: dict, args: dict = {},
                quantities: list = [], extra_args: dict = {}) -> bool:

    classy = state["module"]

    try:
        cosmo = classy.Class()
        cosmo.set(state["params"] | args)
        cosmo.compute()
    except (classy.CosmoComputationError, classy.CosmoSevereError) as e:
        return False

    for quantity in quantities:
        qpath = quantity.split("/")
        if qpath[0] == "Cl":
            spec = qpath[1]

            cls = cosmo.lensed_cl()
            ell = cls["ell"]
            Cl = cls[spec]
            Dl = Cl * (ell * (ell + 1) / (2 * np.pi))

            state[quantity] = Dl[state[quantity + ".modes"]]
        elif qpath[0] == "unlensed_Cl":
            spec = qpath[1]

            cls = cosmo.raw_cl()
            ell = cls["ell"]
            Cl = cls[spec]
            Dl = Cl * (ell * (ell + 1) / (2 * np.pi))

            state[quantity] = Dl[state[quantity + ".modes"]]

    cosmo.struct_cleanup()
    cosmo.empty()

    return True
