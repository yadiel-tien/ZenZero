import logging
import os
import sys
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
    
    # 1. 控制台 Handler (带颜色)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
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
