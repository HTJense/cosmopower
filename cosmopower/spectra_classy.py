from .parser import YAMLParser
import numpy as np


def initialize(parser: YAMLParser, extra_args: dict = {}) -> dict:
    import classy

    state = {}

    for quantity in parser.quantities:
        state[quantity + ".modes"] = parser.modes(quantity)

    if extra_args.get("auto", False):
        extra_args.pop("auto")
        extra_args = classy_args_interpret(parser, extra_args)
        print(f"Auto-generated class settings: {extra_args}")

    state["code"] = "classy"
    state["module"] = classy
    state["params"] = extra_args
    state["derived"] = {}

    return state


def classy_args_interpret(parser: YAMLParser, extra_args: dict = {}) -> dict:
    extra_args["l_max_scalars"] = extra_args.get("l_max_scalars", 0)
    extra_args["l_max_tensors"] = extra_args.get("l_max_tensors", 0)

    extra_args["lensing"] = "no"
    cl_spec = set([])
    cl_modes = set([])

    for quantity in parser.quantities:
        qpath = quantity.split("/")
        if qpath[0] == "Cl" or qpath[0] == "unlensed_Cl":
            if qpath[1] == "tt" or qpath[1] == "te":
                cl_spec.add("tCl")
            if qpath[1] == "te" or qpath[1] == "ee":
                cl_spec.add("pCl")
            if qpath[0] == "Cl" or qpath[1] == "pp":
                extra_args["lensing"] = "yes"
                cl_spec.add("lCl")

            cl_modes.add("s")
            extra_args["l_max_scalars"] = max(extra_args["l_max_scalars"],
                                              parser.modes(quantity).max())
        elif qpath[0] == "tensor_Cl":
            if qpath[1] == "tt" or qpath[1] == "te":
                cl_spec.add("tCl")
            if qpath[1] == "te" or qpath[1] == "ee":
                cl_spec.add("pCl")

            cl_modes.add("t")
            extra_args["l_max_tensors"] = max(extra_args["l_max_tensors"],
                                              parser.modes(quantity).max())

    if "s" not in cl_modes: extra_args.pop("l_max_scalars")
    if "t" not in cl_modes: extra_args.pop("l_max_tensors")
    extra_args["output"] = " ".join(list(cl_spec))
    extra_args["modes"] = " ".join(list(cl_modes))

    return extra_args

def get_spectra(parser: YAMLParser, state: dict, args: dict = {},
                quantities: list = [], extra_args: dict = {}) -> bool:

    classy = state["module"]

    try:
        cosmo = classy.Class()
        cosmo.set(state["params"] | args)
        cosmo.compute()
    except (classy.CosmoComputationError, classy.CosmoSevereError) as e:
        raise e

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
