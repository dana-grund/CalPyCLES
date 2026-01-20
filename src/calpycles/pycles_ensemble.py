"""An ensemble pf PycCLES samples.

Represents e.g. the prior or posterior ensembles, where self.mean_sample
is an additional run with the mean parameters for comparison.

Can also be the training data for an emulator.

USAGE
ensemble = PyCLESensemble(TEST_CASE)
ensemble.init_samples(n_samples=10)
ensemble.run()
ensemble.plot()


TO DO
- define Distribution class (init by samples, including plot)
- define Gaussian class as chile (init by mean+var, see roses?)
"""

# Standard library
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# Third-party
import numpy as np
from numpy.typing import NDArray

# First-party
from calpycles.DYCOMS_RF01.measurements import MeasurementsDYCOMS_RF01
from calpycles.helpers import create_mfdataset_with_nans
from calpycles.helpers import ensure_path
from calpycles.observable import ObsEnsemble
from calpycles.observable import obs_ens_factory
from calpycles.parameters import ParamEnsemble
from calpycles.pycles_sample import PyCLESsample
from calpycles.pycles_sample import get_sample_class
from calpycles.pycles_sample import wait_to_finish
from calpycles.slurm.helpers import dump_slurm_info
from calpycles.slurm.helpers import make_job_settings
from calpycles.slurm.helpers import move_slurm_files
from calpycles.slurm.helpers import run_array


class PyCLESensemble:
    """An ensemble of PyCLES samples."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        name: str = "ensemble",
        path: str = "./",
        verbose: bool = False,
        case: str = "DYCOMS_RF01",
        test: bool = False,
        n_samples: Optional[int] = None,
        parameter_ranges: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize."""
        # general settings
        self.name = name
        self.path = path
        ensure_path(self.path)  # parent_path in PyCLESsample
        self.verbose = verbose
        self.test = test
        self.loaded = False

        # select test case
        self.case = case
        self.sample_class = get_sample_class(case)

        # initialize parameter ensemble

        self.param_ens = ParamEnsemble(
            name=name,
            path=path,
            verbose=verbose,
            case=case,
            n_samples=n_samples,
            parameter_ranges=parameter_ranges,
        )

        # initialized later
        self._obs_ens: ObsEnsemble
        self._samples: List[PyCLESsample]
        self.mean_sample: PyCLESsample  # sample with mean parameters
        self.default_sample: PyCLESsample  # sample with default parameters
        self.namelist_settings: Optional[Dict] = None  # type: ignore # init_samples

    # shorthands
    param_samples = property(lambda self: self.param_ens.samples)
    dist = property(lambda self: self.param_ens.dist)
    n_samples = property(lambda self: self.param_ens.n_samples)

    @property
    def samples(self) -> List[PyCLESsample]:
        """Return the samples."""
        if not hasattr(self, "_samples"):
            self.init_samples()
        return self._samples

    def init_samples(
        self,
        namelist_settings: Optional[Dict] = None,  # type: ignore
        param_samples: Optional[NDArray[np.float64]] = None,
        constrained: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the samples.

        Default: param_samples are passed in normal unconstrained space (as within DA).
        """
        if param_samples is None:
            assert hasattr(
                self.param_ens, "_samples"
            ), "Initialize parameter ensemble before samples, or pass param_samples!"
        else:
            self.param_ens.set(param_samples, constrained=constrained)
            self.param_ens.save()

        self.namelist_settings = namelist_settings  # might be None

        self._samples = [
            self.init_sample(
                name=str(i),
                parameters=p,
                **kwargs,
            )
            for i, p in enumerate(self.param_ens.samples_constrained)
            # (n_samples, n_params)
        ]

    def init_sample(
        self,
        name: str,
        parameters: NDArray[np.float64],
        **kwargs: Any,
    ) -> PyCLESsample:
        """Initialize a single sample."""
        sample: PyCLESsample = self.sample_class(
            name=name,
            parameters=parameters,
            parent_path=self.path,
            verbose=self.verbose,
            test=self.test,
            namelist_settings=self.namelist_settings,
            **kwargs,
        )
        return sample

    def set_dist(self, **kwargs: Any) -> None:
        """Pass the distribution initialization to the parameter ensemble."""
        self.param_ens.init_dist(**kwargs)

    def run(
        self,
        submit_job: Optional[bool] = True,
        job_settings: Optional[Dict[str, Any]] = None,
        run_mean_and_def: Optional[bool] = False,
        do_wait_and_clean: bool = True,
        **clean_kwargs: Any,
    ) -> None:
        """Run each sample in the ensemble.

        Waits until all samples are finished (checking sequentially).
        job_settings will be attributed to each job in the array, so to each sample.
        job_settings is passed here in slurm syntax, e.g. time="04:00:00".

        ALTERNATIVE: Submit each sample in a job (submit_job=False)
            for sample in self.samples:
                sample.run(
                    submit_job=True,
                    do_wait_to_finish=False,
                )

        SLURM SUMMARY
            > myjobs -j $ID >> slurm.summary
        """
        # write namelists
        for s in self.samples:
            s.write_namelist()

        # do not overwrite
        if self.was_run:
            print(f"Ensemble {self.name} has already been run. Skipping simulation.")

        elif not submit_job:
            for s in self.samples:
                s.run(submit_job=False)
            if run_mean_and_def:
                self.run_default_sample(submit_job=False)
                self.run_mean_sample(submit_job=False)

        else:

            # prepare information for slurm
            cmds = [sample.run_command for sample in self.samples]
            paths = [sample.path for sample in self.samples]
            dump_slurm_info(cmds, paths, self.path)

            # prepare job settings
            sample = self.samples[0]
            job_settings_, max_time_s = make_job_settings(
                name="ens",
                nproc=sample.namelist.nproc,
                mem_in_m=sample.mem_in_m,
                time_in_h=sample.computing_time_in_h,
                n_samples=self.n_samples,
                test=sample.test,
            )

            if job_settings is not None:
                job_settings_.update(job_settings)

            # submit ensemble as job array but dont wait for it to finish here
            run_array(
                path=self.path,
                job_settings=job_settings_,
                verbose=self.verbose,
            )
            if run_mean_and_def:
                self.run_mean_sample(do_wait_to_finish=False)
                self.run_default_sample(do_wait_to_finish=False)

            # wait until all samples are finished
            samples = self.samples
            if run_mean_and_def:
                samples += [self.mean_sample, self.default_sample]

            if do_wait_and_clean:
                wait_to_finish(samples, max_time_s=max_time_s)

                # tidy up the ensemble path from slurm stuff
                move_slurm_files(self.path)

                # clean up each sample
                for sample in self.samples:
                    sample.clean_files(**clean_kwargs)

    @property
    def obs_ens(self) -> ObsEnsemble:
        """Return the observable ensemble."""
        if not hasattr(self, "_obs_ens"):
            self.init_obs_ens()
        return self._obs_ens

    def init_obs_ens(
        self,
        load: bool = False,
    ) -> None:
        """Initialize the observable."""
        self._obs_ens = obs_ens_factory(
            case=self.case,
            namelists=[s.namelist for s in self.samples],
            path=self.path,
            name=self.name,
            load=load,
        )
        for sample in self.samples:
            sample.init_observable()

    def run_mean_sample(self, **kwargs: Any) -> None:
        """Run the sample given by the mean of the parameter ensemble.

        kwargs: do_wait_to_finish, submit_job
        """
        if self.verbose:
            print("Running mean sample.")

        mean = self.param_ens.dist.mu
        if self.param_ens.dist_kwargs["dist_type"] == "uniform":
            # TO DO: avoid here, implement some default in dist
            mean = self.param_ens.dist.to_constrained(mean)
        self.mean_sample = self.init_sample(name="mean", parameters=mean)

        self.mean_sample.run(**kwargs)

    def run_default_sample(self, **kwargs: Any) -> None:
        """Run the sample given by the default namelist parameters.

        kwargs: do_wait_to_finish, submit_job
        """
        if self.verbose:
            print("Running default sample.")

        self.default_sample = self.init_sample(
            name="default", parameters=self.param_ens.dist_kwargs["defaults"]
        )

        self.default_sample.run(**kwargs)

    def save_fields(self) -> None:
        """Save fields of all samples in one netcdf file."""
        ds = create_mfdataset_with_nans(
            [s.fields_file for s in self.samples],
        )
        ds.to_netcdf(os.path.join(self.path, "samples_fields.nc"))

    def plot(self, data: Optional[MeasurementsDYCOMS_RF01] = None) -> None:
        """Plot the ensembles of parameters and observations."""
        self.param_ens.plot()
        if hasattr(self, "obs_ens"):
            self.obs_ens.plot(data=data)

    def plot_samples(
        self,
        samples: Union[List[int], int],
        field_variables: Optional[Union[list[str], str]] = None,
    ) -> None:
        """Plot selected ensemble members into their folders.

        Input:
            samples = 3 # plot first 3 samples
            samples = [0,1,2] # plot samples 0, 1, 2

        Output:
            Field plots
            Observable plots
        """
        if isinstance(samples, int):
            idcs = list(range(samples))
        else:
            idcs = samples

        for i in idcs:
            sample = self.samples[i]

            # plot fields
            sample.plot(variables=field_variables)

    @property
    def was_run(self) -> bool:
        """Detect whether all samples have been run."""
        return hasattr(self, "_samples") and all(s.was_run for s in self.samples)

    @property
    def can_load(self) -> bool:
        """Check whether there are parameter samples to load."""
        return self.param_ens.can_load

    def load(self, **kwargs: Any) -> None:
        """Load ensemble from files."""
        self.loaded = True

        if not self.can_load:
            print(
                f"Can't load ensemble {self.name} (the parameter ensemble can't load)."
            )
        else:
            # load parameter ensemble
            self.param_ens.load()

            # check if there are samples to be loaded
            sample_dirs = [
                p
                for p in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, p))
                and p.isnumeric()  # excludes e.g. slurm/ and mean/
            ]
            n_samples = len(sample_dirs)
            if n_samples > 0:
                # load samples
                sample_dirs.sort(key=int)  # sort by number

                # load samples (namelists)
                self._samples = []
                if self.verbose:
                    print(f"Loading {n_samples} samples for ensemble {self.name}...")
                for sample_dir in sample_dirs:
                    sample = self.sample_class(
                        parent_path=self.path,
                        name=sample_dir,  # "0", "1", ...
                        verbose=False,
                        test=self.test,
                        load=True,
                        **kwargs,
                    )

                    self._samples.append(sample)

                # save namelist settings
                self.namelist_settings = self._samples[0].namelist.namelist.copy()
                del self.namelist_settings["meta"]  # simname, uuid

                if self.verbose:
                    print(f"... done loading samples for ensemble {self.name}.")

                # observations
                if self.was_run:
                    self.init_obs_ens(load=True)

            elif os.path.exists(os.path.join(self.path, "samples_fields.nc")):
                if self.verbose:
                    print(
                        "The data is saved in a single sample_fields.nc."
                        "Loading from this format is not implemented."
                    )
            else:
                if self.verbose:
                    print(
                        f"Did not find any samples to load for ensemble {self.name}. "
                        f"However, the parameter ensemble in {self.name} has "
                        f"{self.param_ens.n_samples} samples."
                    )

        self.load_mean_and_default()

    def load_mean_and_default(self) -> None:
        """Load mean and default samples if they exist."""
        if "mean" in os.listdir(self.path):
            self.mean_sample = self.sample_class(
                parent_path=self.path,
                name="mean",
                verbose=self.verbose,
                test=self.test,
            )
            self.mean_sample.load()
            self.mean_sample.init_observable()

            if self.verbose:
                print("Loaded mean sample for ensemble ", self.name)

        if "default" in os.listdir(self.path):
            self.default_sample = self.sample_class(
                parent_path=self.path,
                name="default",
                verbose=self.verbose,
                test=self.test,
            )
            self.default_sample.load()
            self.default_sample.init_observable()

            if self.verbose:
                print("Loaded default sample for ensemble ", self.name)
