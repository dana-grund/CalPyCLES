# CalPyCLES: Ensemble-based calibration of PyCLES

This package was developed for the publication 'Bayesian Calibration of a Large-Eddy Resolving Model towards Campaign Measurements with an Ensemble Kalman Smoother', Grund and Schemm, 2026. This codebase is meant to allow reproducing  the results. We hope bits and pieces of this code help or inspire similar projects. Please contact Dana Grund (dgrund@ethz.ch) in case of any questions.

The package was developed on the Euler machine at ETH Zürich and uses the slurm job scheduling system for ensemble simulations. The setup was left as-is in order to ensure reproducibility on the Euler machine.

The main code snippets for the ensemble-based LES calibration are the following:

- `src/calpycles/enkf_calibration.py`: EnKS, partial updates
- `src/calpycles/observable.py`: Observation operator (extraction of simulated observations from the raw simulation data)
- `src/calpycles/namelist_defaults/DYCOMS_RF01.in`: Default configuration of the simulation ("nature")


## Installation and related packages

Please use the script `environment/setup_pip_env_euler.sh -h`. The script will take care of creating a pip environment called `calpycles` and installing calpycles in it as editable package. The installation can be tested in `test/test.ipynb` on some small test data.


### PyCLES
If you wish to also run the PyCLES simulation yourself, git clone the release of Grund and Schemm 2026, based on Pressel et. al 2015, via `git clone git@github.com:/dana-grund/pycles.git --branch v0.1.0`. Then copy the path you cloned it into `environment/PYCLES_PATH.txt` and use `environment/setup_pip_env_euler.sh -c` to compile PyCLES into the `calpycles` environment using slurm.


### DAPPER
The EnKS implementation in `src/calpycles/enkf_calibration.py` and the functionalities in `src/calpycles/dapper/` are taken from the DAPPER package in `github.com/nansencenter/DAPPER` (Raanes et al. 2024). They are hard-copied and modified, so no installation of the package is required.





## Data and figures for publication Grund and Schemm 2026

The folder `experiments` contains the notebooks used to create data, execute experiments, and plot results of the publication. The simulation data and figures are collected in `experiments/data/` and `experiments/figs/`, respectively. Code for data loading is provided in `data_loaders.py`, while the notebooks rely on the `calpycles` code.

### Obtaining the data
The data can be obtained from the ETH Research Collection repository, http://hdl.handle.net/20.500.11850/791158. In addition to the WENO/FULLRES, LOWRES, and CENTRAL models described in the main manuscript, this repository contains data and results also for the MIXED model mentioned in the supplementary information. As a reference for computational complexity, output files of the WENO/FULLRES nature run are saved in `slurm_output_WENO_FULLRES_nature`.

### Raw data
For each sample (nature sample, posterior mode sample, and samples within an ensemble), the following are available:
- `DYCOMS_RF01_xxx.in`: Input file, containing the entire PyCLES setup including the variable parameter inputs and initial random seed. The PyCLES simulation can be repeated exactly from this input file.
- `Stats.nc`: Statistics computed by PyCLES online, where only those relevant to the publication are kept

For selected samples, there are also:
- `CondStats.nc`: Spectra statistics
- `Fields.nc`: Three-dimensional fields at the final simulation time

In case a simulation did not finish, partial statistics are available in
- `Stats.DYCOMS_RF01_XXX.nc`

### Processed data
For single samples (nature sample, posterior mode sample), the following are available:
- `samples_observations.nc`: Post-processed simulated observations in the format plotted in the publication (see file structure below)

For ensembles, the following are available:
- `samples_parameters_constrained.nc`: Parameter samples in physical space
- `samples_parameters_unconstrained.nc`: Noramlized parameter samples
- `samples_observations.nc`: Post-processed simulated observations in the format plotted in the publication (see file structure below)
- `samples_observations_full_profiles.nc`: as above, with exact profile time steps instead of averages
- `samples_observations_full_timeseries.nc`: as above, with exact time steps instead of linear fits
- `ensemble_stats/`: pre-computed ensemble statistics of the mean, median, and confidence intervals of the profiles and timeseries as plotted


All post-processed observation files have the following structure:
    
    netcdf samples_observations {
    dimensions:
            sample = 1 ;
            z = 21 ;
            fit = 2 ;
    variables:
            double w_mean2(sample, z) ;
                    w_mean2:_FillValue = NaN ;
                    w_mean2:coordinates = "moment" ;
            double w_mean3(sample, z) ;
                    w_mean3:_FillValue = NaN ;
                    w_mean3:coordinates = "moment" ;
            double qt_mean(sample, z) ;
                    qt_mean:_FillValue = NaN ;
                    qt_mean:coordinates = "moment" ;
            double ql_mean(sample, z) ;
                    ql_mean:_FillValue = NaN ;
                    ql_mean:coordinates = "moment" ;
            double thetali_mean(sample, z) ;
                    thetali_mean:_FillValue = NaN ;
                    thetali_mean:coordinates = "moment" ;
            int64 sample(sample) ;
            double z(z) ;
                    z:_FillValue = NaN ;
            int64 moment ;
            double cloud_top_ts_fit(fit, sample) ;
                    cloud_top_ts_fit:_FillValue = NaN ;
                    cloud_top_ts_fit:coordinates = "moment" ;
            double cloud_base_ts_fit(fit, sample) ;
                    cloud_base_ts_fit:_FillValue = NaN ;
                    cloud_base_ts_fit:coordinates = "moment" ;
            int64 fit(fit) ;
    }






## Development tools

As this package was created with the APN Python blueprint, it comes with a stack of development tools, which are described in more detail on (https://meteoswiss-apn.github.io/mch-python-blueprint/). Here, we give a brief overview on what is implemented.


### Workflow for committing

Activate the environment:

    source environment/setup_env_pip_euler.sh

Run the pre-commit checks:

    pre-commit run  # over all staged changes
    pre-commit run -a  # over all changes
    pre-commit run --file file.py  # over a single file

Do NOT modify the code while the checks are running, or abort the checks - you risk losing changes! Note that a full pre-commit check may take a while (some minutes), in particular the `pylint` and `mypy` parts.

Commit your work:

    git commit [-m "message"]  # runs pre-commit and commits in case of no errors
    git-commit [-m "message"] --no-verify  # commits without pre-commit


### Testing and coding standards

Testing your code and compliance with the most important Python standards is a requirement for Python software written in APN. To make the life of package administrators easier, the most important checks are run automatically on GitHub actions. If your code goes into production, it must additionally be tested on CSCS machines, which is only possible with a Jenkins pipeline (GitHub actions is running on a GitHub server).


### Pre-commit on GitHub actions

`.github/workflows/pre-commit.yml` contains a hook that will trigger the creation of your environment (unpinned) on the GitHub actions server and
then run various formatters and linters through pre-commit. This hook is only triggered upon pushes to the main branch (in general: don't do that)
and in pull requests to the main branch.


### Jenkins

A jenkinsfile is available in the `jenkins/` folder. It can be used for a multibranch jenkins project, which builds
both commits on branches and PRs. Your jenkins pipeline will not be set up
automatically. If you need to run your tests on CSCS machines, contact DevOps to help you with the setup of the pipelines. Otherwise, you can ignore the jenkinsfiles
and exclusively run your tests and checks on GitHub actions.






## Credits and references

This package was created with [`copier`](https://github.com/copier-org/copier) and the [`MeteoSwiss-APN/mch-python-blueprint`](https://meteoswiss-apn.github.io/mch-python-blueprint/) project template.

Grund, D. & Schemm, S. (2026). Bayesian Calibration of a Large-Eddy Resolving Model towards Campaign Measurements with an Ensemble Kalman Smoother. In preparation.

Pressel, K. G., Kaul, C. M., Schneider, T., Tan, Z., & Mishra, S. (2015). Large-eddy simulation in an anelastic framework with closed water and entropy balances. Journal of Advances in Modeling Earth Systems, 7 (3), 1425–1456. doi: 10.1002/2015MS000496

Raanes, P. N., Chen, Y., & Grudzien, C. (2024). DAPPER: Data Assimilation with Python: A Package for Experimental Research. Journal of Open Source Software, 9 (94), 5150. doi: 10.21105/joss.05150



