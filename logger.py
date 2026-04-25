import logging
import sys
import inspect


def getLogger(name=None, level=logging.INFO,
              fmt='%(levelname)s - %(filename)s:%(lineno)d'):
    """
    获取带行号的logger（最简单版本）

    用法：
        logger = getLogger(__name__)  # 一行代码搞定
        logger.info("日志信息")

    Args:
        name: logger名称，默认为调用模块的__name__
        level: 日志级别
        fmt: 日志格式，默认包含行号

    Returns:
        配置好的logger
    """

    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', '__main__')

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger
