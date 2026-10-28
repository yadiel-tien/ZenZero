import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler
from core.utils.config import CONFIG

# 颜色配置 (用于控制台输出)
class ColorFormatter(logging.Formatter):
    """自定义颜色格式化器"""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

def get_logger(name: str, level=logging.INFO, log_to_file=True) -> logging.Logger:
    """
    获取标准化 Logger
    :param name: logger名称 (推荐使用 __name__ 或业务模块名)
    :param level: 日志级别
    :param log_to_file: 是否保存到文件
    """
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    
    # 1. 控制台 Handler（带颜色）——仅当 stdout 真连着终端。
    # 颜色是给"人眼+tty"的装饰层：训练子进程 stdout 被 runner 重定向进 console.log 时
    # 不是 tty，若照吐转义码，装饰色会烙进记录文件（train 整行蓝、与 latest.log 两样）。
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter() if sys.stdout.isatty() else logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s'))
    logger.addHandler(console_handler)

    # 2. 文件 Handler (按天滚动)
    if log_to_file:
        log_dir = os.path.join(CONFIG['log_dir'], name.split('.')[0]) # 按大模块分类
        os.makedirs(log_dir, exist_ok=True)
        
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s')
        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, 'latest.log'),
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 避免日志向上传递给 root logger 导致重复打印
    logger.propagate = False
    
    return logger


def install_thread_excepthook(logger_name: str = 'crash') -> logging.Logger:
    """把"任何线程里未捕获的异常"写进日志文件。

    为什么需要：Python 默认只把线程异常打到 stderr，而训练是后台跑的——日志文件里
    什么都看不到，线程悄悄死掉，请求要等到 300s 超时才暴露（8/27 引擎投递线程
    `deque mutated` 就是这样静默死的；10/27 用户报"evaluate 报错"时日志同样干净）。
    挂上后 stderr 照旧打印，同时落一份带完整栈到 logs/crash/latest.log。
    """
    logger = get_logger(logger_name)

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        logger.error('未捕获的线程异常（线程已死，相关请求可能挂到超时）: %s',
                     getattr(args.thread, 'name', '?'), exc_info=(args.exc_type, args.exc_value,
                                                                args.exc_traceback))

    threading.excepthook = _hook
    return logger
