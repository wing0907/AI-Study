import logging

def setup_logger(level=logging.INFO):
    logger = logging.getLogger("lawai")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger
