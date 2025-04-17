

import argparse
import multiprocessing
import os
import random
from functools import partial
from multiprocessing import cpu_count

from genbio.aidosp.msa_retrieve.msar.runner import Runner
from genbio.aidosp.msa_retrieve.msar.utils.general import seq_encoder
from genbio.aidosp.msa_retrieve.msar.utils.io_utils import load_yaml, read_fasta
from genbio.aidosp.msa_retrieve.msar.utils.logger import Logger

logger = Logger.logger


def run_seq_group_alignments(input, dir_code, file_code, alignment_runner, output_dir):
    alignment_dir = os.path.join(output_dir, dir_code, file_code)

    try:
        os.makedirs(alignment_dir, exist_ok=True)
        if input.endswith(".fasta") or input.endswith(".fa"):
            cmd = f"cp {input} {alignment_dir}"
            os.system(cmd)
            base_filename = os.path.basename(input)
            cmd = f"mv {alignment_dir}/{base_filename} {alignment_dir}/raw.fasta"
            os.system(cmd)
            fasta_path = f"{alignment_dir}/raw.fasta"
        else:
            seq = input.upper()  # missing residue is lowercase.
            hash_code = seq_encoder(seq)
            line = f">{hash_code}\n{seq}\n"
            fasta_path = f"{alignment_dir}/raw.fasta"
            with open(fasta_path, "w") as file:
                file.write(line)
        alignment_runner.run(fasta_path, alignment_dir)
    except Exception as e:
        logger.warning(e)
        logger.error(f"Failed when running {alignment_dir}. Skip this case.")
        return 0

    return 1


def parse_and_align(instances, alignment_runner, args):
    for instance in instances:
        if instance.endswith(".fasta") or instance.endswith(".fa"):
            fasta_path = os.path.join(args.input, instance)
            with open(fasta_path) as infile:
                sequences, descriptions = read_fasta(infile.read())
                infile.close()
            # file_name = os.path.splitext(f)[0]
            file_code = seq_encoder(sequences[0])
            input = fasta_path
        else:
            file_code = seq_encoder(instance)
            input = instance
        dir_code = file_code[0:2]
        run_seq_group_alignments(
            input, dir_code, file_code, alignment_runner, args.output_dir
        )


def str_none(input):
    if input is None or input.lower() == "none":
        return None
    else:
        return input


def main(args, config):
    alignment_runner = Runner(config, no_cpus=args.cpus_per_task)

    if args.input.endswith(".fasta") or args.input.endswith(".fa"):
        with open(args.input) as infile:
            instances, descriptions = read_fasta(infile.read())
            infile.close()
    else:
        instances = list(os.listdir(args.input))
    args.no_tasks = min(args.no_tasks, len(instances))
    logger.info(f"no_tasks={args.no_tasks}")

    if args.shuffle_file_list:
        random.shuffle(instances)

    def split_up_arglist(arglist):
        t_arglist = []
        for i in range(args.no_tasks):
            t_arglist.append(arglist[i :: args.no_tasks])

        return t_arglist

    func = partial(
        parse_and_align,
        alignment_runner=alignment_runner,
        args=args,
    )
    task_arglist = [[a] for a in split_up_arglist(instances)]

    ps = []
    for i, task_args in enumerate(task_arglist):
        logger.info(f"Started process {i}...")
        p = multiprocessing.Process(target=func, args=task_args)
        ps.append(p)
        p.start()

    for p in ps:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="""Path to directory containing FASTA or one fasta file containing a lot of sequences""",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory in which to output alignments"
    )
    parser.add_argument("--config_yaml_path", type=str, default=None)
    parser.add_argument(
        "--raise_errors",
        action="store_true",
        default=False,
        help="Whether to crash on parsing errors",
    )
    parser.add_argument(
        "--cpus_per_task", type=int, default=cpu_count(), help="Number of CPUs to use"
    )
    parser.add_argument(
        "--no_tasks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--shuffle_file_list",
        action="store_true",
        default=True,
        help="Whether to shuffle file list",
    )

    args = parser.parse_args()

    logger.info(f"config_yaml_path={args.config_yaml_path}")
    config = load_yaml(args.config_yaml_path)

    main(args, config)

    logger.info("End.")
