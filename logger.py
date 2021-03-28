import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[1m"
    blue = "\x1b[1;34m"
    yellow = '\x1b[33m'
    red = '\x1b[31m'
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s][%(name)s][%(levelname)s]: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class Logger():
    def __init__(self, name):
        # create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel("DEBUG")
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel("DEBUG")
        # create formatter
        formatter = CustomFormatter()

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def log_debug(self, msg):
        self.logger.debug(msg)

    def log_info(self, msg):
        self.logger.info(msg)

    def log_warning(self, msg):
        self.logger.warning(msg)

    def log_error(self, msg):
        self.logger.error(msg)



if __name__ == "__main__":
    log = Logger("a")
    log.log_info("check text")
    log.log_warning("check text war")
    log.log_error("check text err")