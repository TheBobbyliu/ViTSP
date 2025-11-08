import logging

def _configure_logging(filename='logger') -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(message)s",
                        datefmt="%H:%M:%S",
                        force=True,
                        filename='logs/' + filename + '.log')
