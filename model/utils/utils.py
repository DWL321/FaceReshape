import logging
import os
from datetime import datetime
from functools import partial
from subprocess import DEVNULL, check_call, check_output

from tqdm import tqdm as std_tqdm

from model.utils.log_utils import TEDIUM, log

tqdm = partial(std_tqdm, bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}")


def datetime_str():
    return f"{datetime.now():%Y%m%d_%H%M%S}"


def run(cmd, env=os.environ, logging_level=logging.DEBUG):
    log(f"[run] {cmd}", logging_level)
    check_call(cmd, shell=True, env=env, stdout=DEVNULL, stderr=DEVNULL)


def get_output(cmd, env, logging_level=TEDIUM):
    log(f"[run] {cmd}", logging_level)
    return check_output(cmd, shell=True, env=env, stderr=DEVNULL)
