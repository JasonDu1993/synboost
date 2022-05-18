# -*- coding: utf-8 -*-
# @Time    : 2022/5/11 17:34
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : log_util.py
# @Software: PyCharm
import logging


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "openselfsup".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(__name__.split('.')[0])  # i.e., openselfsup
    # logger = logging.getLogger()  # i.e., openselfsup
    # if the logger has been initialized, just return it这里注释了下面两行，因为一直为True，后面的代码根本调用不到
    if len(logger.handlers):
        return logger

    format_str = '[%(asctime)s %(name)s %(levelname)s] %(message)s'
    if isinstance(log_level, str) or isinstance(log_level, int):
        log_level = str(log_level)
        if log_level.upper() == "NOTSET" or log_level == "0":
            log_level = logging.NOTSET
        elif log_level.upper() == "DEBUG" or log_level == "10":
            log_level = logging.DEBUG
        elif log_level.upper() == "INFO" or log_level == "20":
            log_level = logging.INFO
        elif log_level.upper() == "WARNING" or log_level.upper() == "WARN" or log_level == "30":
            log_level = logging.WARNING
        elif log_level.upper() == "ERROR" or log_level == "40":
            log_level = logging.ERROR
        elif log_level.upper() == "CRITICAL" or log_level.upper() == "FATAL" or log_level == "50":
            log_level = logging.CRITICAL
        else:
            raise Exception(
                "only support log level NOTSET DEBUG INFO WARNING ERROR CRITICAL, but now is {}".format(log_level))
    # 之前通过basicConfig就可以改变logger的有效级别，通过getEffectiveLevel可以查看对应的有效等级，但这里没有成功需要自己在设置一遍才行logger.setLevel(log_level)
    logging.basicConfig(format=format_str, level=log_level)
    logger.setLevel(log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger
