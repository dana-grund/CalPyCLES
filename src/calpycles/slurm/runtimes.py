"""Diagnose runtimes of job arrays from summary file.

EACH BLOCK LOOKS LIKE

Job information
 Job ID                          : 26232239_0
 Job name                        : LH_s
 Status                          : COMPLETED
 Running on node                 : eu-a2p-380
 User                            : dgrund
 Shareholder group               : ls_math
 Slurm partition (queue)         : normal.24h
 Command                         : sbatch -J LH_s --parsable -n 32 --array=[0-63] -A ls_math --output=/cluster/work/climate/dgrund/working_dir/25-03-14_dycoms_new_ens/latHyp_N64_small//slurm/slurm_%A_%a.out --mem-per-cpu=500M --time=24:00:00 --constraint=ibfabric6|ibfabric7 --wrap=python /cluster/work/climate/dgrund/git/dana-grund/CalPyCLES/src/calpycles/scripts/run_jobarray/run_jobarray.py -p /cluster/work/climate/dgrund/working_dir/25-03-14_dycoms_new_ens/latHyp_N64_small/ --infile_path /cluster/work/climate/dgrund/working_dir/25-03-14_dycoms_new_ens/latHyp_N64_small//infiles/ --JOBID $SLURM_ARRAY_TASK_ID --nproc 32 -c DYCOMS_RF01 --pycles /cluster/work/climate/dgrund/git/dana-grund/pycles_worktree/DYCOMS_RF01/
 Working directory               : /cluster/work/climate/dgrund/working_dir/25-03-14_dycoms_new_ens/latHyp_N64_small
Requested resources
 Requested runtime               : 1-00:00:00
 Requested cores (total)         : 32
 Requested nodes                 : 1
 Requested memory (total)        : 16000 MiB
Job history
 Submitted at                    : 2025-03-14T15:45:03
 Started at                      : 2025-03-14T16:16:46
 Queue waiting time              : 31 m 43 s
Resource usage
 Wall-clock                      : 02:50:08
 Total CPU time                  : 3-17:43:58
 CPU utilization                 : 98.89%
 Total resident memory           : 10360.81 MiB
 Resident memory utilization     : 64.75%

SAMPLE USAGE

    N_JOBS = 64

    JOB_ID = 26232239
    ENS_DIR = "/cluster/work/climate/dgrund/working_dir/25-03-14_dycoms_new_ens/latHyp_N64_small"
    make_summary(JOB_ID, N_JOBS, ENS_DIR)

    JOB_ID = 26232402
    ENS_DIR = "/cluster/work/climate/dgrund/working_dir/25-03-14_dycoms_new_ens/latHyp_N64_large"
    make_summary(JOB_ID, N_JOBS, ENS_DIR)


"""

# Standard library
import argparse
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# mypy: ignore-errors
# flake8: noqa
# noqa
# pylint: skip-file

SUMMMARY_SCRIPT = "/cluster/work/climate/dgrund/git/dana-grund/CalPyCLES/src/calpycles/slurm/print_job_array_summary.sh"


def make_summary(JOB_ID, N_JOBS, ENS_DIR):
    """Save summary of the job array to a file."""
    slurm_dir = os.path.join(ENS_DIR, "slurm")

    # print job summary
    os.system(f"cd {slurm_dir}; bash {SUMMMARY_SCRIPT} {JOB_ID} {N_JOBS}")
    file = f"job_array_summary_{JOB_ID}.txt"

    # read run times
    times = []
    nproc = []
    with open(os.path.join(slurm_dir, file)) as f:
        # read all lines and pick those that say wall-clock
        lines = f.readlines()
        for line in lines:
            if "Wall-clock" in line:
                # print(line)
                time = line.split(" : ")[1].strip().split(":")
                times.append(
                    (int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])) / 3600  # h
                )

            # get nproc
            if "Requested cores (total)" in line:
                nproc.append(int(line.split(":")[1].strip()))

    nproc = np.array(nproc)
    times = np.array(times)

    n = nproc[0]
    assert np.all(nproc == n), "Number of processors changed during job array!"

    # plot histogram
    plt.figure()
    plt.hist(times, bins=20)
    plt.xlabel("Wall-clock time")
    plt.ylabel("Frequency")
    plt.title(f"Wall-clock time histogram (nproc={n}; mean={np.mean(times):2.2f} h)")
    file = os.path.join(slurm_dir, f"hist_wall_clock{JOB_ID}.png")
    plt.savefig(file)
    print(f"Saved wall-clock histogram to {file}")
    print(f"Mean wall-clock time: {np.mean(times)}")

    # plot histogram
    plt.figure()
    plt.hist(times * n, bins=20)
    plt.xlabel("CPU time")
    plt.ylabel("Frequency")
    plt.title(
        f"CPU time histogram (obtained with nproc={n}; mean={np.mean(times*n):2.2f} h)"
    )
    file = os.path.join(slurm_dir, f"hist_cpu_{JOB_ID}.png")
    plt.savefig(file)
    print(f"Saved CPU histogram to {file}")
    print(f"Mean CPU time: {np.mean(times*n)}")


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--job_id", type=int, required=True)
    args.add_argument("--n_jobs", type=int, required=True)
    args.add_argument("--ens_dir", type=str, required=True)
    args = args.parse_args()

    make_summary(args.job_id, args.n_jobs, args.ens_dir)
