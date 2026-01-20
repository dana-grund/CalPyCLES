r"""Interface to run ensembles as slurm job arrays.

Will be called by PyCLESensemble as
> python {array_script} --cmds_path {cmds_path}
                        --paths_path {paths_path}
                        --index \\$SLURM_ARRAY_TASK_ID
"""

# Standard library
import argparse
import os

run_job_in_array_script = os.path.abspath(__file__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmds_path",
        type=str,
        required=True,
        help="Path containing the 0.txt files that contain the commands to run.",
    )
    parser.add_argument(
        "--paths_path",
        type=str,
        required=True,
        help="Path containing the 0.txt files that contain the samplt path.",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default=0,
        help="The index of the command to run in this array.",
    )
    args = parser.parse_args()

    index = int(args.index)

    # choose command for this index
    with open(os.path.join(args.cmds_path, f"{index}.txt"), "r", encoding="utf-8") as f:
        cmd = f.readline()

    # choose path for this index
    with open(
        os.path.join(args.paths_path, f"{index}.txt"), "r", encoding="utf-8"
    ) as f:
        path = f.readline()

    os.chdir(path)
    os.system(cmd)
