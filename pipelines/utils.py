"""
Helpers for the univariate->multivariate pipeline.

* submit_job_and_wait: submit an LSF (bsub) job and block until it finishes.
* create_yml_files:     expand one base config into a per-disease config file each.

These helpers are LSF-specific (they shell out to `bsub`/`bjobs`).
"""
import subprocess, time, re, os, logging
import yaml, copy, argparse

logger = logging.getLogger(__name__)


def submit_job_and_wait(bsub_cmd: list, wait_time=10):
    """Submit an LSF job and poll `bjobs` until it reaches DONE or EXIT.

    Raises RuntimeError if the submission output does not contain a job ID
    (previously this raised an opaque AttributeError on `None.group`).
    """
    # Submit job and capture job ID
    result = subprocess.run(bsub_cmd, capture_output=True, text=True)
    m = re.search(r"<(\d+)>", result.stdout or "")
    if m is None:
        raise RuntimeError(
            "Could not parse an LSF job ID from bsub output. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    jobid = m.group(1)
    logger.info(f"Submitted job {jobid}, waiting...")

    # Sleep briefly so the job is registered with the scheduler before we poll.
    time.sleep(wait_time)

    # Poll until job finishes
    while True:
        bjobs = subprocess.run(["bjobs", jobid], capture_output=True, text=True)
        lines = bjobs.stdout.strip().splitlines()

        # A transient scheduler hiccup can yield no status line; retry rather than crash.
        if len(lines) < 2:
            logger.warning(f"No status returned for job {jobid}, retrying...")
            time.sleep(wait_time)
            continue

        # Column order can vary by LSF site/config, so detect the status token by value
        # rather than by a fixed column index (the previous fixed index was fragile).
        fields = lines[1].split()
        LSF_STATES = {"PEND", "RUN", "DONE", "EXIT", "PSUSP", "USUSP",
                      "SSUSP", "WAIT", "ZOMBI", "UNKWN"}
        stat = next((f for f in fields if f in LSF_STATES), "")
        if stat in ("DONE", "EXIT"):
            logger.info(f"stat={stat}, job {jobid} finished")
            break
        else:
            logger.info(f"stat={stat or 'UNKNOWN'}, job {jobid} still running or pending")
        time.sleep(wait_time)


def create_yml_files(base_yml: str, diseases: list, save_path: str):
    """
    Create yml files for each disease, then save it to a directory
    """
    os.makedirs(save_path, exist_ok=True)
    with open(base_yml, "r") as f:
        config = yaml.safe_load(f)
        orig_disease = config["inputs"]["disease_name"]

    for disease in sorted(diseases):
        if orig_disease[0] == "all": pass
        elif disease not in orig_disease: continue
        new_config = copy.deepcopy(config)
        new_config["inputs"]["disease_name"] = [disease]

        with open(f"{save_path}/{disease}.yml", 'w') as file:
            yaml.dump(new_config, file, default_flow_style=False)


def main(args):
    """
    Run this script if you want to populate yml files across diseases
    """
    # Run code to create yml files
    diseases = os.listdir(args.disease_path)
    diseases = [i.split(".")[0] for i in diseases]
    create_yml_files(args.yml_full_path, diseases, f"{args.save_path}/{args.save_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Save path")
    parser.add_argument("--save_name", type=str, help="Save name")
    parser.add_argument("--yml_full_path", type=str, help="Full path to yml file")
    parser.add_argument("--disease_path", type=str, help="Disease path")
    args = parser.parse_args()
    main(args)
    