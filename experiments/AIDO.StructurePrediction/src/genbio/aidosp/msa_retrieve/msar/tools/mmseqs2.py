
import glob
import subprocess
from typing import Any, Mapping

from genbio.aidosp.msa_retrieve.msar.tools import utils
from genbio.aidosp.msa_retrieve.msar.utils.logger import Logger

logger = Logger.logger


class MMSeqs:
    def __init__(
        self,
        config: dict[str, Any],
        binary_path: str,
        database_path: str,
    ):
        self.binary_path = binary_path
        self.database_path = database_path
        self.config = config

        logger.info(database_path)
        if not glob.glob(database_path + "_*"):
            logger.error("no database %s", database_path)
            raise ValueError(f"no database {database_path}")

    def generate_search_cmd(self, binary_path, database_path, base_path, out_a3m):
        cmd = f"{binary_path} search {base_path}/qdb {database_path} {base_path}/res {base_path}/tmp {self.config.search}"
        cmd += f";{binary_path} expandaln {base_path}/qdb {database_path}.idx {base_path}/res {database_path}.idx {base_path}/res_exp {self.config.expandaln}"
        cmd += (
            f";{binary_path} mvdb {base_path}/tmp/latest/profile_1 {base_path}/prof_res"
        )
        cmd += f";{binary_path} lndb {base_path}/qdb_h {base_path}/prof_res_h"
        cmd += f";{binary_path} align {base_path}/prof_res {database_path}.idx {base_path}/res_exp {base_path}/res_exp_realign {self.config.align}"
        cmd += f";{binary_path} filterresult {base_path}/qdb {database_path}.idx {base_path}/res_exp_realign {base_path}/res_exp_realign_filter {self.config.filter}"
        cmd += f";{binary_path} result2msa {base_path}/qdb {database_path}.idx {base_path}/res_exp_realign_filter {base_path}/{out_a3m}  {self.config.result2msa}"
        cmd += f";head {base_path}/{out_a3m}"
        cmd += f";{binary_path} rmdb {base_path}/res_exp_realign_filter"
        cmd += f";{binary_path} rmdb {base_path}/res_exp_realign"
        cmd += f";{binary_path} rmdb {base_path}/res_= exp"
        cmd += f";{binary_path} rmdb {base_path}/res"
        return str(cmd)

    def generate_cmd(
        self,
        input_fasta_path,
        binary_path,
        database_path,
        base_path="tmp_base",
        out_a3m="mmdb_out.a3m",
    ):
        cmd = "export MMSEQS_CALL_DEPTH=1"
        cmd += f";{binary_path} createdb {input_fasta_path} {base_path}/qdb"
        cmd += "; " + self.generate_search_cmd(
            binary_path, database_path, base_path, out_a3m
        )
        # delete temporary files
        cmd += f" ;{binary_path} rmdb {base_path}/qdb"
        cmd += f" ;{binary_path} rmdb {base_path}/qdb_h"
        cmd += f" ;{binary_path} rmdb {base_path}/res"
        return (
            cmd,
            f"{base_path}/{out_a3m}",
        )

    def query(self, input_fasta_path: str, msa_out_path=None) -> Mapping[str, Any]:
        with utils.tmpdir_manager() as query_tmp_dir:
            (cmd, out_a3m_path) = self.generate_cmd(
                input_fasta_path,
                self.binary_path,
                self.database_path,
                base_path=query_tmp_dir,
            )
            logger.info('Launching subprocess "%s"', cmd)
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            with utils.timing("MMseqs query"):
                stdout, stderr = process.communicate()
                retcode = process.wait()

            logger.info(f"retcode={retcode}")
            if retcode:
                # Logs have a 15k character limit.
                for error_line in stderr.decode("utf-8").splitlines():
                    if error_line.strip():
                        logger.error(error_line.strip())
                raise RuntimeError(
                    "MMseqs failed\nstdout:\n{}\n\nstderr:\n{}\n".format(stdout.decode("utf-8"), stderr[:500_000].decode("utf-8"))
                )

            with open(out_a3m_path) as f:
                a3m = f.read()

            raw_output = {
                "a3m": a3m,
                "output": stdout,
                "stderr": stderr,
            }
        return raw_output
