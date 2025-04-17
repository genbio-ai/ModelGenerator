
import contextlib
import os

from fold.utils.logger import Logger

logger = Logger.logger

def make_lock_file(lock_file_path):
    "Create a lock ."
    f = open(lock_file_path, "x")
    _ip = get_ip()
    stat = "_".join([str(os.getpid()).strip(), _ip])
    f.write(stat)
    f.close()


def get_ip():
    """Get IP addr of the current machine..."""
    return ",".join(
        os.popen(
            "ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d addr:"
        )
        .read()
        .strip()
        .split("\n")
    )


def pid_issame(lock_pid, custom_pid):
    """Compare the pid and custom_pid."""
    _ip = get_ip()
    custom_stat = "_".join([custom_pid.strip(), _ip])
    if lock_pid == custom_stat:
        return True
    else:
        return False


def get_lock_owner(lock_file_path):
    """Get the owner."""
    try:
        with open(lock_file_path, "r") as f:
            pid = f.readline().strip()
    except FileNotFoundError as e:
        pid = "-3"
    return pid


@contextlib.contextmanager
def lock_manager(lock_file_path):
    """Context manager that deletes a lock.temp file on exit."""
    lock_file = os.path.join(lock_file_path, "lock.temp")
    try:
        try:
            make_lock_file(lock_file)
            yield get_lock_owner(lock_file)
        except FileExistsError as e:
            logger.info(
                f"The folder has been locked {lock_file_path}. Exception={str(e)}"
            )
            yield get_lock_owner(lock_file)
        except Exception as e:
            logger.info(f"Catch unknown exception. Exits. Exception={str(e)}")
            yield "-1"
    finally:
        pass