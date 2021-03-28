from logger import Logger


class Base:
    def __init__(self, name):
        self.logger = Logger(name=name)

    def log_debug(self, msg):
        self.logger.logger.debug(msg)

    def log_info(self, msg):
        self.logger.logger.info(msg)

    def log_warning(self, msg):
        self.logger.logger.warning(msg)

    def log_error(self, msg):
        self.logger.logger.error(msg)
