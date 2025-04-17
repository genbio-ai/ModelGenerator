
import os
from multiprocessing import cpu_count

from genbio.aidosp.msa_retrieve.msar.tools import mmseqs2, mmseqs2_api
from genbio.aidosp.msa_retrieve.msar.utils.logger import Logger

logger = Logger.logger


def run_msa_tool(
    msa_runner,
    fasta_path: str,
    msa_out_path: str,
    msa_format: str,
    max_sequences: int | None = None,
):
    logger.info(f"[begin] retrieve msa {msa_out_path}")
    if msa_format == "sto" and max_sequences is not None:
        result = msa_runner.query(fasta_path, max_sequences, msa_out_path=msa_out_path)
    else:
        result = msa_runner.query(fasta_path, msa_out_path=msa_out_path)

    if result:
        assert msa_out_path.split(".")[-1] == msa_format
        with open(msa_out_path, "w") as f:
            f.write(result[msa_format])

        logger.info(f"[end] retrieve msa {msa_out_path}")

    return result


class Runner:
    def __init__(
        self,
        config,
        no_cpus: int | None = None,
    ):
        self.config = config
        if no_cpus is None:
            no_cpus = cpu_count()

        self.msa_runner_list, self.output_list, self.msa_format_list = (
            [],
            [],
            [],
        )
        self.max_sequences_list = []

        if config.tools.mmseqs2.enable:
            # self.config_list = list()
            dbs = []
            db_name_list = config.tools.mmseqs2.dbs.split(",")
            for db_name in db_name_list:
                dbs.append(config.data[db_name].database_path)
                runner = mmseqs2.MMSeqs(
                    config=config.tools.mmseqs2[db_name],
                    binary_path=config.tools.mmseqs2.binary_path,
                    database_path=config.data[db_name].database_path,
                )
                self.msa_runner_list.append(runner)
                self.output_list.append(f"mmdb_{db_name}_hits.a3m")
                self.msa_format_list.append("a3m")
                self.max_sequences_list.append(None)

        if config.tools.mmseqs2_api.enable:
            runner = mmseqs2_api.MMSeqsAPI(
                config=config.tools.mmseqs2_api,
            )
            self.msa_runner_list.append(runner)
            self.output_list.append(None)
            self.msa_format_list.append("a3m")
            self.max_sequences_list.append(None)

    def run(
        self,
        fasta_path: str,
        output_dir: str,
    ):
        num_runners = len(self.msa_runner_list)
        for i in range(num_runners):
            if self.output_list[i]:
                out_path = os.path.join(output_dir, self.output_list[i])
                if os.path.exists(out_path):
                    logger.info(f"{out_path} is existed.")
                    continue
            else:
                out_path = output_dir

            logger.info(f"search for out_path={out_path}")
            run_msa_tool(
                msa_runner=self.msa_runner_list[i],
                fasta_path=fasta_path,
                msa_out_path=out_path,
                msa_format=self.msa_format_list[i],
                max_sequences=self.max_sequences_list[i],
            )
