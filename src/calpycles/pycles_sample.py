"""A single PyCLES simulation.

# python 3.12: kwarg annotation. So far, **kwargs are avoided.
from typing import TypedDict
from typing import Unpack
class SampleKwargs(TypedDict):
    parent_path: str
    name: str
    verbose: bool
    namelist_file: Optional[str]
"""

# Standard library
import json
import os
import time
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# Third-party
import numpy as np
import xarray as xr
from numpy.typing import NDArray

# First-party
from calpycles.helpers import ensure_path
from calpycles.helpers import process_camlab_style
from calpycles.namelist import Namelist
from calpycles.namelist import namelist_factory
from calpycles.observable import ObsEnsemble
from calpycles.observable import ObsEnsembleDYCOMS_RF01
from calpycles.observable import obs_factory
from calpycles.parameters import Parameters
from calpycles.parameters import parameter_classes
from calpycles.plotting.plot_3d_fields import plot_3D_fields
from calpycles.slurm.helpers import make_job_settings
from calpycles.slurm.helpers import run_single


class PyCLESsample(ABC):
    """A single PyCLES simulation."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        parent_path: str = "./",
        name: str = "0",  # sample number / nature / mean / ...
        verbose: bool = False,
        test: bool = False,
        namelist_settings: Optional[Dict[str, Any]] = None,
        parameters: Optional[NDArray[np.float64]] = None,
        load: Optional[bool] = None,
        do_clean: bool = True,
    ) -> None:
        """Initialize."""
        self.name = name
        self.verbose = verbose
        self.test = test
        self.wrote_namelist = False
        self.do_clean = do_clean

        self.path = os.path.join(parent_path, name)
        ensure_path(parent_path)
        ensure_path(self.path)

        # --- properties
        self._fields: xr.Dataset
        self._namelist: Namelist
        self._observable: ObsEnsemble
        self._parameters: Parameters

        # --- attributes required in child classes
        self.case: str
        self.computing_time_in_h: float
        self.mem_in_m: float

        # --- namelist and uuid
        if load and self.can_load:
            self.wrote_namelist = True
            self.load()  # load uuid
        else:
            self.init_namelist(namelist_settings)  # create uuid

        # --- parameters
        if parameters is not None:
            self.parameters.set(parameters)

    @property
    def parameter_class(self) -> Type["Parameters"]:
        """Get the Parameter class for this case."""
        return parameter_classes(self.case)

    @property
    def simname(self) -> str:
        """Get the simulation name."""
        return self.case + "_" + self.name

    def init_namelist(
        self,
        namelist_settings: Optional[Dict] = None,  # type: ignore
        make_uuid: bool = False,
    ) -> None:
        """Write default namelist to file."""
        self._namelist = namelist_factory(
            case=self.case,
            test=self.test,
            path=self.path,
            simname=self.simname,
            make_uuid=make_uuid,
        )
        self.update_namelist(namelist_settings)

    def update_namelist(
        self,
        namelist_settings: Optional[Dict] = None,  # type: ignore
    ) -> None:
        """Update and overwrite namelist."""
        # initialize in case is not
        _ = self.namelist

        if namelist_settings is not None:
            self._namelist.update(namelist_settings)

    def write_namelist(self) -> None:
        """Write namelist to file."""
        self.namelist.write()
        self.wrote_namelist = True

        if self.verbose:
            print(f"Wrote namelist to: {self.path}")

    @property
    def namelist(self) -> Namelist:
        """Get the namelist."""
        if not hasattr(self, "_namelist"):
            self.init_namelist()
            assert isinstance(self._namelist, Namelist), "Namelist init went wrong."
            if self.verbose:
                print("Initializing default namelist.")

        return self._namelist

    @property
    def parameters(self) -> Parameters:
        """Get the class managing the parameters.

        Set specific parameter values as
        > self.parameters.set(values: NDArray[np.float64])
        """
        if not hasattr(self, "_parameters"):
            assert hasattr(self, "_namelist"), "Initialize namelist before parameters!"
            self._parameters = self.parameter_class(self.namelist)

        return self._parameters

    @property
    def observable(self) -> ObsEnsemble:
        """The observable."""
        if not hasattr(self, "_observable"):
            self.init_observable()
        return self._observable

    def init_observable(self) -> None:
        """Initialize the observable."""
        self._observable = obs_factory(
            case=self.case,
            namelist=self.namelist,
            path=self.path,
            name=self.name,
        )

    @property
    def was_run(self) -> bool:
        """Detect whether the simulation was run."""
        return os.path.exists(self.namelist.out_dir)

    @property
    def has_finished(self) -> bool:
        """Check if the simulation has finished."""
        finished_cleaned = os.path.exists(self.stats_file)
        finished_uncleaned = os.path.exists(self.namelist.stats_file_pycles_format)
        return finished_cleaned or finished_uncleaned

    @property
    def can_load(self) -> bool:
        """Check if exactly one namelist is found."""
        namelist_files = [f for f in os.listdir(self.path) if f.endswith(".in")]
        return len(namelist_files) == 1

    def load(
        self,
    ) -> None:
        """Load namelist from file."""
        if not self.can_load:
            if self.verbose:
                print(f"WARNING: Can't load namelist from {self.path}")
                print("Continue without loading, using default namelist instead.")
            return

        self.wrote_namelist = True
        namelist_files = [f for f in os.listdir(self.path) if f.endswith(".in")]
        namelist_file = os.path.join(self.path, namelist_files[0])
        with open(namelist_file, "r", encoding="utf-8") as fh:
            namelist_loaded = json.load(fh)

        # overwrite all namelist entries, including uuid
        self.init_namelist(namelist_settings=namelist_loaded, make_uuid=False)

        if self.verbose:
            print(f"Loaded namelist from {namelist_file}")

    def run(
        self,
        submit_job: bool = True,
        do_wait_to_finish: bool = True,
        force: bool = False,
        clean_kwargs: Optional[Any] = None,
    ) -> None:
        """Run the simulation."""
        # be very conservative with overwriting

        if not self.wrote_namelist:
            self.write_namelist()

        # in case parameters are NaN, skip the simulation
        if self.parameters.is_nan:
            # pretend the run was tried and failed, such that self.was_run is True
            if not os.path.exists(self.namelist.out_dir):
                os.mkdir(self.namelist.out_dir)
            print(f"Sample {self.name} has NaN parameters, skipping simulation.")
            return

        # do not overwrite existing results
        if self.was_run:
            if force:
                print(f"Sample {self.name} was already run, forcing to overwrite.")
            else:
                print(f"Sample {self.name} was already run, skipping simulation.")
                return

        # else, simulate
        if submit_job:
            job_settings, max_time_s = make_job_settings(
                name=self.name,
                nproc=self.namelist.nproc,
                mem_in_m=self.mem_in_m,
                time_in_h=self.computing_time_in_h,
                test=self.test,
            )
            run_single(
                path=self.path,
                cmd=self.run_command,
                job_settings=job_settings,
                verbose=self.verbose,
            )
            if do_wait_to_finish:
                wait_to_finish(self, max_time_s=max_time_s)
        else:
            cwd = os.getcwd()
            os.chdir(self.path)

            print(f"Waiting for {self.stats_file}...")

            outfile = os.path.join(self.path, "out.out")
            os.system(self.run_command + " > " + outfile)

            assert self.has_finished
            print(f"\t... simulation {self.name} finished successfully.")

            os.chdir(cwd)

        # clean up files (can only clean once the simulation has finished)
        if self.do_clean and do_wait_to_finish:
            if clean_kwargs is None:
                clean_kwargs = {}
            self.clean_files(**clean_kwargs)

    def clean_files(
        self,
        delete_fields: bool = True,
        select_stats: bool = True,
        delete_cond_stats: bool = True,
        delete_second_infile: bool = True,
        remove_uuid_from_outdir: bool = True,
    ) -> None:
        """Clean after running."""
        # pylint: disable=unused-argument
        # keep kwargs for child classes
        return

    @property
    def run_command(self) -> str:
        """The command to execute in order to run the simulation."""
        # read PYCLES_PATH from a file containing "PYCLES_PATH="
        calpycles_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        pycles_path_file = os.path.join(calpycles_path, "environment", "PYCLES_PATH.txt")
        with open(pycles_path_file, "r", encoding="utf-8") as file:
            pycles_path = file.read().strip()[12:]
        pycles = os.path.join(pycles_path, "main.py")

        # assemble the command
        namelist_file = self.namelist.file
        nproc = self.namelist.nproc
        if nproc == 1:
            cmd = f"python {pycles} {namelist_file}"
        else:
            cmd = f"mpirun -np {nproc} python {pycles} {namelist_file}"

        return cmd

    @property
    def stats_file(self) -> str:
        """Get the final output file."""
        return self.namelist.files["stats_file"]

    @property
    def cond_stats_file(self) -> str:
        """Get the final output file."""
        return self.namelist.files["cond_stats_file"]

    @property
    def fields_file(self) -> str:
        """Get the final output file."""
        return self.namelist.files["fields_file"]

    @property
    def fields(self) -> xr.Dataset:
        """Load the fields of all written time steps."""
        assert (
            self.has_finished
        ), f"Simulation '{self.name}' did not finish, could not find {self.stats_file}"

        if not hasattr(self, "_fields"):
            self._fields = xr.load_dataset(self.fields_file)

        return self._fields

    @abstractmethod
    def plot(
        self,
        variables: Optional[Union[list[str], str]] = None,
        times: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> None:
        """Define plotting depending on the test case.

        TO DO:
            - Find good uniform kwargs
        """
        raise NotImplementedError("Has to be defined in child class.")


class SampleDYCOMS_RF01(PyCLESsample):
    """A single PyCLES simulation with DYCOMS RF01 settings."""

    # pylint: disable=invalid-name

    case = "DYCOMS_RF01"
    computing_time_in_h = 24  # 50 cpuh (estimate for 1 core) # assume >= 4 cores
    mem_in_m = 6000

    def clean_files(
        self,
        delete_fields: bool = True,
        select_stats: bool = True,
        delete_cond_stats: bool = True,
        delete_second_infile: bool = True,
        remove_uuid_from_outdir: bool = True,
        camlab_style: bool = False,
    ) -> None:
        """After running, keep only what is needed for the observable."""
        # pylint: disable=too-many-locals

        print(f"Cleaning sample files of {self.name}...")
        files = []  # files kept

        # remove uuid to avoid confusion (hard-coded in Namelist!)
        if remove_uuid_from_outdir:
            namelist = self.namelist
            namelist_dict = namelist.namelist
            uuid = namelist_dict["meta"]["uuid"]
            out_dir_orig = os.path.join(
                namelist.path,
                namelist_dict["output"]["output_root"],
                "Output." + namelist.simname + "." + uuid[-5:],
            )
            if os.path.exists(out_dir_orig):
                os.rename(out_dir_orig, namelist.out_dir)

        # delete fields file
        if os.path.exists(self.fields_file):
            if delete_fields:
                # print(f"Deleting fields file {self.fields_file}")
                os.remove(self.fields_file)
            else:
                files.append(self.fields_file)

        # select stats variables (deleting reference)
        if os.path.exists(self.stats_file):
            if select_stats:
                # print(f"Selecting variables in stats file {self.stats_file}")
                profiles = xr.open_dataset(self.stats_file, group="profiles")
                timeseries = xr.open_dataset(self.stats_file, group="timeseries")

                profiles = profiles[ObsEnsembleDYCOMS_RF01.profile_variables]
                timeseries = timeseries[ObsEnsembleDYCOMS_RF01.timeseries_variables]

                os.remove(self.stats_file)

                profiles.to_netcdf(self.stats_file, group="profiles")
                timeseries.to_netcdf(self.stats_file, group="timeseries", mode="a")

                profiles.close()
                timeseries.close()

        # delete spectra
        cond_stats_file = self.cond_stats_file
        if os.path.exists(cond_stats_file):
            if delete_cond_stats:
                # print(f"Deleting spectra file {cond_stats_file}")
                os.remove(cond_stats_file)
            else:
                files.append(cond_stats_file)

        # delete copy of .in file in output directory
        second_infile = os.path.join(self.namelist.out_dir, self.namelist.filename)
        if delete_second_infile and os.path.exists(second_infile):
            # print(f"Deleting second infile {second_infile}")
            os.remove(second_infile)

        if camlab_style:
            # deletes group reference from stats_file!
            process_camlab_style(
                self.fields_file,
                self.stats_file,
                cond_stats_file,
            )

    def plot(
        self,
        variables: Optional[Union[list[str], str]] = None,
        times: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> None:
        """Plot fields, profiles, timeseries."""
        # 3D fields
        if os.path.exists(self.fields_file):
            plot_3D_fields(
                outdir=self.namelist.out_dir,
                case=self.case,
                times=times,
                savefolder=self.path,
                savename="",
            )

        # observations: profiles and timeseries
        if hasattr(self, "observable"):
            self.observable.plot()


def get_sample_class(case: str = "DYCOMS_RF01") -> Type["PyCLESsample"]:
    """Return the corresponding sample class, like a factory."""
    if case == "DYCOMS_RF01":
        return SampleDYCOMS_RF01

    raise NotImplementedError(
        f"The case '{case}' is not implemented as a PyCLESsample!"
    )


def wait_to_finish(
    samples: Union[List[PyCLESsample], PyCLESsample], max_time_s: float
) -> None:
    """Wait until all samples are finished of max_time_s s have passed."""
    if not isinstance(samples, list):
        samples = [samples]

    waiting_interval = 10  # s
    time_elapsed = 0  # s

    # check if all samples are finished
    print("Waiting for samples to finish...")
    while not all(s.has_finished for s in samples):
        time.sleep(waiting_interval)
        time_elapsed += waiting_interval
        if time_elapsed > max_time_s:
            print(
                (
                    "WARNING: Timeout while waiting for simulation(s) "
                    "to finish. Continuing."
                )
            )
            break

    # Check on single sample
    if len(samples) == 1:
        s = samples[0]
        if s.has_finished:
            print(f"Simulation {s.name} finished successfully.")
        elif s.was_run:
            print(f"Simulation {s.name} was run, but not finished.")
        else:
            print(f"Simulation {s.name} was not run.")

    # Check on ensemble
    else:
        not_finished = [s for s in samples if not s.has_finished]
        not_run = [s for s in samples if not s.was_run]

        if len(not_finished) == 0:
            print("All samples finished.")
        elif len(not_run) > 0:
            print(f"WARNING: The following samples have not been run: {not_run}")
        else:
            print(f"WARNING: The following samples have not finished: {not_finished}")
