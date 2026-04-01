from .parser import YAMLParser
import numpy as np


def initialize(parser: YAMLParser, extra_args: dict = {}) -> dict:
    import classy

    state = {}

    for quantity in parser.quantities:
        state[quantity + ".modes"] = parser.modes(quantity)

    extra_args["l_max_scalars"] = extra_args.get("l_max_scalars", 0)
    extra_args["l_max_tensors"] = extra_args.get("l_max_tensors", 0)

    cl_modes = []

    for quantity in parser.quantities:
        if quantity.split("/")[0] == "Cl":
            cl_modes.append("s")
            extra_args["l_max_scalars"] = max(extra_args["l_max_scalars"],
                                              parser.modes(quantity).max())
        if quantity.split("/")[0] == "tensor_Cl":
            cl_modes.append("t")
            extra_args["l_max_tensors"] = max(extra_args["l_max_tensors"],
                                              parser.modes(quantity).max())

    if "s" not in cl_modes: extra_args.pop("l_max_scalars")
    if "t" not in cl_modes: extra_args.pop("l_max_tensors")
    extra_args["modes"] = " ".join(cl_modes)

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
        print(str(e))
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
        elif qpath[0] == "tensor_Cl":
            spec = qpath[1]

            cls = cosmo.raw_cl()
            ell = cls["ell"]
            Cl = cls[spec]
            Dl = Cl * (ell * (ell + 1) / (2 * np.pi))

            state[quantity] = Dl[state[quantity + ".modes"]]

    cosmo.struct_cleanup()
    cosmo.empty()

    return True
