#!/bin/bash
#
# Manage pip environment (conda is discouraged on Euler)
#
# By Dana Grund after Euler got updated to Ubuntu (2024)
# Based on setup_env.sh by Stefan Rüdisühli (2022)
#

# Default options
DEFAULT_ENV_NAME="calpycles"
ENV_NAME="${DEFAULT_ENV_NAME}"
ACTIVATE=true
REMOVE=false
UPDATE=false
EXPORT=false
INSTALL=false
COMPILE=false
HELP=false

help_msg="Usage: bash $(basename "${0}") [-n NAME] [-u] [-r] [-a] [-e] [-i] [-c] [-h]

This script manages a pip environment on Euler,
with fixed environment module versions:

module load stack/2024-04 # includes gcc/8.5.0
module load openmpi/4.1.6
module load python/3.9.18
module load hdf5/1.14.3
module load netcdf-c/4.9.2

Default: (Create and) activate the env.
Full setup: -ei
Full setup including pycles compilation: -eic
To activate the env in your shell, use "source"

Options:
 -n NAME    Env name [default: ${DEFAULT_ENV_NAME}]
 -a         Activate environment [default: true]
 -r         Remove environment and create it from scratch [default: false]
 -u         Update environment according requirements.txt [default: false]
 -e         Export environment files by pip freeze  [default: false]
 -i         Install project into env as editable [default: false]
 -c         Compile PyCLES inside the environment [default: false]
 -h         Print this help message and exit
"

# Eval command line options
while getopts n:arueicdh flag; do
    case ${flag} in
        n) ENV_NAME=${OPTARG};;
        r) REMOVE=true;;
        a) ACTIVATE=true;;
        u) UPDATE=true
           EXPORT=true
           ;;
        e) EXPORT=true;;
        i) INSTALL=true;;
        c) COMPILE=true;;
        h) HELP=true
           ACTIVATE=false
           ;;
        ?) HELP=true
           ACTIVATE=false
           ;;
    esac
done

# Help message
if ${HELP}; then
    echo "${help_msg}"
else
    # Load environment modules
    module load stack/2024-04 # includes gcc/8.5.0
    module load openmpi/4.1.6
    module load python/3.9.18
    module load hdf5/1.8.21
    # module load hdf5/1.14.3 # seems not to be lined to openmpi?
    module load netcdf-c/4.9.2
    module list
fi

# Directories
THIS_DIR=$(pwd)
THIS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$(dirname $THIS_SCRIPT_DIR)
GIT_DIR=$(dirname $PROJECT_DIR)
cd ${THIS_SCRIPT_DIR}

# Environment
VENV_DIR=${THIS_SCRIPT_DIR}/.venv/${ENV_NAME}
VENV_ACT=${VENV_DIR}/bin/activate

# Remove environment
if ${REMOVE}; then
    rm -rf ${VENV_DIR}
    echo Removed venv ${ENV_NAME}.
fi

# Create environment
if [ ! -d "${VENV_DIR}" ]; then
    echo Creating virtual environment at ${VENV_DIR}
    python -m venv --system-site-packages $VENV_DIR
    source $VENV_ACT
    pip install --upgrade pip
    pip install -r requirements.txt
    echo Created venv ${ENV_NAME} and installed requirements.txt.
    python -m ipykernel install --user --name calpycles
    EXPORT=true
fi

# Update environment
if ${UPDATE}; then
    source $VENV_ACT
    pip install --upgrade -r requirements.txt
    echo Updated venv ${ENV_NAME} according to requirements.txt.
fi

# Activate environment
if ${ACTIVATE}; then
    source $VENV_ACT
    echo Activated venv ${ENV_NAME}.
fi

# Export environment
if ${EXPORT}; then
    source $VENV_ACT
    pip freeze > environment.txt
    echo Exported venv ${ENV_NAME} to environment.txt.
fi

# Install current project into new env as editable
if ${INSTALL}; then
    cd ${PROJECT_DIR}
    python -m pip install -e . --no-deps
    echo Installed the project into the venv ${ENV_NAME} as editable.
    cd ${THIS_SCRIPT_DIR}
fi

# Compile PyCLES inside the environment
if ${COMPILE}; then
    echo Compiling PyCLES inside the venv ${ENV_NAME} in a job.
    echo Check for success in ./compile_pycles.out, .err:
    echo The .out file should end with "Finished combining ranks per time step."
    sbatch compile_pycles_slurm_script.sh
fi

# Go back to where we came from
cd ${THIS_DIR}
