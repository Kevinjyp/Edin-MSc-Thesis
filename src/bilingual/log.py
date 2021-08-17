import logging
import time

def get_logger(    
        LOG_FORMAT     = '%(asctime)s [%(levelname)s]: %(message)s',
        LOG_NAME       = '',
        STREAM         = True,
    ):
    LOG_NAME = f'{LOG_NAME}.{time.strftime("%Y%m%d%H", time.localtime())}'

    log           = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(LOG_FORMAT)

    # comment this to suppress console output
    if STREAM:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        log.addHandler(stream_handler)

    file_handler_info = logging.FileHandler(f'{LOG_NAME}.log', mode='w')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_debug = logging.FileHandler(f'{LOG_NAME}.debug', mode='w')
    file_handler_debug.setFormatter(log_formatter)
    file_handler_debug.setLevel(logging.DEBUG)
    log.addHandler(file_handler_debug)

    file_handler_error = logging.FileHandler(f'{LOG_NAME}.err', mode='w')
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    log.setLevel(logging.DEBUG)

    return log