"""Class managing the setup of a namelist to be taken as .in file for PyCLES."""

# Standard library
import json
import os
import pprint
import uuid
from typing import Any
from typing import Dict
from typing import Tuple

# Third-party
from flatten_dict import flatten  # type: ignore
from flatten_dict import unflatten

# First-party
from calpycles.DYCOMS_RF01.fluxes import get_theta_from_T
from calpycles.DYCOMS_RF01.fluxes import lh_flux
from calpycles.DYCOMS_RF01.fluxes import sh_flux

this_path = dir_path = os.path.dirname(os.path.realpath(__file__))


class Namelist:
    """A single Namelist."""

    def __init__(
        self,
        case: str = "DYCOMS_RF01",
        test: bool = False,
        simname: str = "None",
        path: str = "./",
        verbose: bool = False,
        make_uuid: bool = True,
    ) -> None:
        """Initialize."""
        self.verbose = verbose

        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

        case_str = case
        if test:
            case_str = case + "_test"
        default = os.path.join(this_path, "namelist_defaults", case_str + ".in")
        assert os.path.exists(default), f"Default namelist not found: {default}"

        self.namelist: Dict  # type: ignore
        with open(default, "r", encoding="utf-8") as fh:
            self.namelist = json.load(fh)

        if simname != "None":
            self.update({"meta": {"simname": simname}})

        if make_uuid:
            self.create_uuid()

    def update(self, new_namelist: Dict) -> None:  # type: ignore
        """Update namelist with new values.

        The structure should be as in namelist.
        Maximal three nested layers:
        {"sgs": {"UniformViscosity": {"diffusivity": 75.0}}}
        """
        old = flatten(self.namelist)
        new = flatten(new_namelist)
        old.update(new)

        self.namelist = unflatten(old)

        if self.verbose:
            print(f"Updated namelist {self.simname} with {new_namelist}.")

    def create_uuid(self) -> None:
        """Create a unique identifier for the simulation."""
        self.namelist["meta"]["uuid"] = str(uuid.uuid4())

    def write(self) -> None:
        """Write .in file to be read by PyCLES."""
        namelist = self.namelist
        file = self.file
        assert (
            namelist["meta"]["simname"] is not None
        ), "Simname not specified in namelist dictionary!"
        # assert not os.path.exists(file),  \
        # f"Namelist exists, not overwriting: {file}"

        with open(file, "w", encoding="utf-8") as fh:
            if self.verbose:
                pprint.pprint(namelist)
            json.dump(namelist, fh, sort_keys=True, indent=4)

    def __repr__(self) -> str:
        """print(Namelist) should print self.namelist."""
        return pprint.pformat(self.namelist)

    @property
    def nproc(self) -> int:
        """Number of cores used."""
        nproc: int = (
            self.namelist["mpi"]["nprocx"]
            * self.namelist["mpi"]["nprocy"]
            * self.namelist["mpi"]["nprocz"]
        )
        return nproc

    @property
    def filename(self) -> str:
        """Name of the .in file."""
        return self.simname + ".in"

    @property
    def file(self) -> str:
        """Path to the .in file."""
        return os.path.join(self.path, self.filename)

    @property
    def simname(self) -> str:
        """Name of the simulation."""
        return self.namelist["meta"]["simname"]  # type: ignore

    @property
    def out_dir(self) -> str:
        """Where the simulation files are stored.

        Originally, PyCLES assigns output directories with the uuid:
            uuid_ = self.namelist["meta"]["uuid"]
            return str(
                os.path.join(
                    self.path, output_root, "Output." + self.simname + "." + uuid_[-5:]
                )
            )
        """
        output_root = self.namelist["output"]["output_root"]
        if output_root.startswith("./"):
            output_root = output_root[2:]
        return os.path.join(self.path, output_root, "Output." + self.simname)

    @property
    def stats_file_pycles_format(self) -> str:
        """Path to the stats file in PyCLES format before any cleaning."""
        output_root = self.namelist["output"]["output_root"]
        uuid_ = self.namelist["meta"]["uuid"]
        out_dir = os.path.join(
            self.path, output_root, "Output." + self.simname + "." + uuid_[-5:]
        )
        return os.path.join(out_dir, "Stats.nc")

    @property
    def files(self) -> Dict[str, str]:
        """Directories derived from the namelist content."""
        namelist = self.namelist

        assert namelist["postprocessing"][
            "collapse_folders"
        ], "Please use namelist['postprocessing']['collapse_folders']=True."
        # if not, directories can be found in PyCLES.PostProcessing.pyx.

        fields_file = os.path.join(self.out_dir, "Fields.nc")
        stats_file = os.path.join(self.out_dir, "Stats.nc")
        cond_stats_file = os.path.join(self.out_dir, "CondStats.nc")

        return {
            "fields_file": fields_file,
            "stats_file": stats_file,
            "cond_stats_file": cond_stats_file,
        }


class NamelistDYCOMSRF01(Namelist):
    """Namelist for the DYCOMS RF01 case (fixed meas. of profiles and timeseries)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize."""
        super().__init__(case="DYCOMS_RF01", **kwargs)

    def update(self, new_namelist: Dict) -> None:  # type: ignore
        """Update namelist with new values.

        Same as in Namelist, adding the flux computation
        in case other parameters are given.
        """
        # pylint: disable=too-many-locals

        def get_input(group: str, name: str) -> Tuple[bool, float]:
            if group in new_namelist and name in new_namelist[group]:
                return True, new_namelist[group][name]
            if name in self.namelist[group]:
                return False, self.namelist[group][name]
            raise ValueError(f"Missing input: {name}")

        u1, sst = get_input("surface", "sst")
        u2, cm = get_input("surface", "cm")
        u3, ug = get_input("forcing", "ug")
        u4, qtg = get_input("initial", "qtg")

        # compute and add fluxes
        do_update_fluxes = u1 or u2 or u3 or u4
        if do_update_fluxes:

            # translate T into thetal
            have_tg = "initial" in new_namelist and "tg" in new_namelist["initial"]
            assert have_tg, "Need tg to compute fluxes."
            tg: float = new_namelist["initial"]["tg"]
            new_namelist["initial"].update(
                {
                    "thetal_g": get_theta_from_T(tg),
                    "tg": tg,  # not needed by PyCLES but added for clarity
                }
            )

            if "surface" not in new_namelist:
                new_namelist["surface"] = {}
            new_namelist["surface"].update(
                {
                    "ft": sh_flux(sst=sst, T=tg, cm=cm, u=ug),
                    "fq": lh_flux(sst=sst, qtg=qtg, cm=cm, u=ug),
                }
            )

        # update namelist
        old = flatten(self.namelist)
        new = flatten(new_namelist)
        old.update(new)
        self.namelist = unflatten(old)

        if self.verbose:
            print(f"Updated namelist {self.simname} with {new_namelist}.")


def namelist_factory(case: str = "DYCOMS_RF01", **kwargs: Any) -> Namelist:
    """Select the namelist corresponding to the test case."""
    if case == "DYCOMS_RF01":
        return NamelistDYCOMSRF01(**kwargs)
    raise ValueError(f"Unknown case to create a Namelist: {case}")
