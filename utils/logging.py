import logging
import os


LOG_FORMAT  = '%(asctime)s [%(levelname)s] %(message)s'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


class LogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(LOG_FORMAT, TIME_FORMAT)

    def format(self, record):
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        n = len(record.asctime) + len(record.levelname) + 4
        record.message = record.getMessage().replace("\n", "\n" + " " * n)
        s = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)
        return s
    

def set_logger(name, logfile=None, level=logging.INFO, mode='a'):
    logger = logging.getLogger(name)
    console_hdl = logging.StreamHandler()
    console_hdl.setFormatter(LogFormatter())
    logger.addHandler(console_hdl)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        file_hdl = logging.FileHandler(logfile, mode=mode)
        file_hdl.setFormatter(LogFormatter())
        logger.addHandler(file_hdl)
    logger.setLevel(level)
    return logger