from calpycles.pycles_ensemble import PyCLESensemble
from calpycles.pycles_sample import SampleDYCOMS_RF01
from calpycles.enkf_calibration import EnKFcalibration
from calpycles.DYCOMS_RF01.measurements import SynthMeasurementsDYCOMS_RF01
from calpycles.plotting import COLORS

import os

DATA_FOLDER = "./data/"

# --- model properties

model_names = [
    "WENO_FULLRES",
    "WENO_LOWRES",
    "MIXED_FULLRES",
    "CENTRAL_FULLRES",
]

# C_PRIOR = COLORS[0]
# C_WENO_FULLRES = COLORS[1]
# C_MIXED = COLORS[2]
# C_CENTRAL = COLORS[3]
# C_WENO_LOWRES = COLORS[4]
# C_WENO_FULLRES_SYNTH = COLORS[5]

colors = {
    "PRIOR": COLORS[0],
    "WENO_FULLRES": COLORS[1],
    "MIXED_FULLRES": COLORS[2],
    "CENTRAL_FULLRES": COLORS[3],
    "WENO_LOWRES": COLORS[4],
    "SYNTH": COLORS[5],
}

scalar_orders = [5, 5, 5, 6]
momentum_orders = [5, 5, 6, 6]
full_res = [True, False, True, True]


def make_namelist_settings(model_name):
    i = model_names.index(model_name)
    
    if full_res[i]:
        dx = 35.
        dz = 5.
    else:
        dx = 50.
        dz = 15.
    
    Lx = 35 * 96
    Lz = 5 * 300
    
    nx = int(Lx/dx)
    nz = int(Lz/dz)

    return {
        "mpi": {
            "nprocx": 8,
            "nprocy": 1,
            "nprocz": 1
        },
        "grid": {
            "dims": 3,
            "dx": dx,
            "dy": dx,
            "dz": dz,
            "gw": 3,
            "nx": nx,
            "ny": nx,
            "nz": nz
        },
        "fields_io": {
            "frequency": 14400  # write hourly fields for nature runs: 3600
        },
        "momentum_transport": {
            "order": momentum_orders[i],
        },
        "scalar_transport": {
            "order": scalar_orders[i],
        },
    }

# --- loaders 


def load_ens(model_name="WENO_FULLRES", i_seed=0):
    """Load an ensemble.
    
    model_name: str
        Name of the model: WENO_FULLRES, WENO_LOWRES, MIXED_FULLRES, CENTRAL_FULLRES
    """
    if model_name == "WENO_FULLRES":
        name = f"DYCOMS_RF01_N64_seed{i_seed}"
    else:
        name = f"DYCOMS_RF01_N64"

    ens = PyCLESensemble(
        name=name,
        path=os.path.join(DATA_FOLDER, model_name, name),
        verbose=True,
        case="DYCOMS_RF01",
        test=False,
        n_samples=64,
    )
    
    ens.load()

    return ens


def load_ens_posterior(model_name="WENO_FULLRES", i_seed=0, synthetic=False):
    """Load the posterior ensemble, if it exists."""
    if model_name == "WENO_FULLRES":
        name = f"DYCOMS_RF01_N64_seed{i_seed}_posterior"
    else:
        name = f"DYCOMS_RF01_N64_posterior"

    if synthetic:
        path = os.path.join(DATA_FOLDER, model_name, "SYNTH", name)
    else:
        path = os.path.join(DATA_FOLDER, model_name, "REAL", name)
    if not os.path.exists:
        raise ValueError(f"Could not find posterior ensemble in {path}")

    ens = PyCLESensemble(
        name=name,
        path=path,
        verbose=True,
        case="DYCOMS_RF01",
        test=False,
        n_samples=64,
        # parameter_ranges=parameter_ranges,
    )
    
    ens.load()

    return ens


def load_nature(model_name="WENO_FULLRES"):
    """Load a single sample."""

    nature = SampleDYCOMS_RF01(
        name="nature",
        parent_path=os.path.join(DATA_FOLDER, model_name),
        load=True,
    )
    
    return nature


def load_mode(model_name="WENO_FULLRES", meas_name=""):
    """Load a single sample."""

    parent_path = os.path.join(DATA_FOLDER, model_name, meas_name)
    sample = SampleDYCOMS_RF01(
        name="mode",
        parent_path=parent_path,
        load=True,
    )
    
    return sample


# --- prepare calibration


def load_da(model_name, synthetic=False, i_seed=0):
    """Initialize the data assimilation."""
    ens = load_ens(model_name, i_seed)

    if synthetic:

        nature = load_nature(model_name)

        synth = SynthMeasurementsDYCOMS_RF01(file=nature.observable.file_xr)

        DA = EnKFcalibration(
            ens,
            synth.observations_np,
            synth.dist_obs_error,
            seed=default_seeds[model_name],
            plot_folder = f"figs/{model_name}/SYNTH",
        )
    
    else:
        DA = EnKFcalibration(
            ens,
            seed=default_seeds[model_name],
            plot_folder = f"figs/{model_name}/REAL",
        )

    return DA


# --- seeds for observation perturbation [TO DO: INCORPORATE IN DA RESULTS]

default_seeds = {
    "WENO_FULLRES": 458758,
    "MIXED_FULLRES": 125698,
    "CENTRAL_FULLRES": 458523,
    "WENO_LOWRES": 254789,
}

default_seeds_synth = {
    model: default_seeds[model] + 1 for model in model_names
}
