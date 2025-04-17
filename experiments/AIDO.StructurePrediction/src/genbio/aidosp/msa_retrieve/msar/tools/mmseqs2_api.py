"""

Modified based on https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py
"""

import os
import random
import tarfile
import time
from typing import Any, Mapping

import requests
from genbio.aidosp.msa_retrieve.msar.tools import utils
from genbio.aidosp.msa_retrieve.msar.utils.io_utils import read_fasta
from genbio.aidosp.msa_retrieve.msar.utils.logger import Logger
from tqdm import tqdm

logger = Logger.logger

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


def run_mmseqs2(
    x,
    prefix,
    msa_out_path,
    use_env=True,
    use_filter=True,
    use_pairing=False,
    pairing_strategy="greedy",
    host_url="https://api.colabfold.com",
    user_agent: str = "",
) -> tuple[list[str], list[str]] | None:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        headers["User-Agent"] = "fold"

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
            logger.info(out)
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching status from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching result from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, _redo = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while _redo:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                logger.info("********************************")
                logger.info(out)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors. "
                        "Please confirm your input is a valid protein sequence. "
                        "If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception(
                        "MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
                    )

                # wait for job to finish
                _id, _time = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(_id)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        _time += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if _time < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - _time))
                    _redo = False

                if out["status"] == "ERROR":
                    _redo = False
                    raise Exception(
                        "MMseqs2 API is giving errors. "
                        "Please confirm your input is a valid protein sequence. "
                        "If error persists, please try again an hour later."
                    )

            # Download results
            download(_id, tar_gz_file)  # pyright: ignore[reportPossiblyUnboundVariable]

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    target_file_list = []
    if use_pairing:
        target_file = f"{msa_out_path}/mmseqs_api_pair.a3m"
        cmd = f"mv {path}/pair.a3m {target_file}"
        os.system(cmd)
        target_file_list.append(target_file)
    else:
        target_file = f"{msa_out_path}/mmseqs_api_uniref.a3m"
        cmd = f"mv {path}/uniref.a3m {target_file}"
        os.system(cmd)
        target_file_list.append(target_file)

        if use_env:
            target_file = f"{msa_out_path}/mmseqs_api_bfd.mgnify30.metaeuk30.smag30.a3m"
            cmd = f"mv {path}/bfd.mgnify30.metaeuk30.smag30.a3m {target_file}"
            os.system(cmd)
            target_file_list.append(target_file)
    logger.info(f"[end] generate {target_file_list}")


class MMSeqsAPI:
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    def query(
        self, input_fasta_path: str, msa_out_path=None
    ) -> Mapping[str, Any] | None:
        logger.info(f"input_fasta_path={input_fasta_path}")
        logger.info(f"msa_out_path={msa_out_path}")

        exists = False
        if self.config.use_pairing:
            if os.path.exists(f"{msa_out_path}/mmseqs_api_pair.a3m"):
                exists = True
        else:
            uniref_exists, env_exists = False, False
            if os.path.exists(f"{msa_out_path}/mmseqs_api_uniref.a3m"):
                uniref_exists = True
            if self.config.use_env:
                if os.path.exists(
                    f"{msa_out_path}/mmseqs_api_bfd.mgnify30.metaeuk30.smag30.a3m"
                ):
                    env_exists = True
            else:
                env_exists = True
            if env_exists and uniref_exists:
                exists = True

        if exists:
            logger.info(f"{msa_out_path} is existed.")
            return None

        with utils.tmpdir_manager() as query_tmp_dir:
            with open(input_fasta_path) as infile:
                sequences, descriptions = read_fasta(infile.read())
                infile.close()
            run_mmseqs2(
                sequences,
                query_tmp_dir,
                msa_out_path,
                use_env=self.config.use_env,
                use_filter=self.config.use_filter,
                use_pairing=self.config.use_pairing,
                pairing_strategy=self.config.pairing_strategy,
            )

        return None
