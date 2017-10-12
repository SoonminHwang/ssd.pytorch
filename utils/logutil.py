import logging
import logging.handlers
import os

class _Config(object):
    log_file = True
    log_console = True
    output_file = None
    
_config = _Config()

_log = logging.getLogger()
_log.setLevel(logging.INFO)

def _setup_logs():
    # setup python logger
    handlers = list(_log.handlers)
    fmt = logging.Formatter('[%(levelname)s] [%(asctime)-11s] %(message)s')
    for h in handlers:
        _log.removeHandler(h)
    
    if _config.log_console:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        _log.addHandler(h)
    
    if _config.log_file and _config.output_file:
        h = logging.FileHandler(_config.output_file)
        h.setFormatter(fmt)
        _log.addHandler(h)

   
_setup_logs()

def set_output_file(filename):    
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)        
    _config.output_file = filename
    _setup_logs()

def get_output_file():
    return _config.output_file if _config.output_file else '.'

def getLogger():
    return _log

def logging_run_info(settings, **kwargs):
    
    for key, value in kwargs.items():
        settings[key] = value   

    _log.info('Exp time: {}'.format(settings['exp_time']))
    for key, value in settings.items():
        if key == 'exp_time':
            continue
        _log.info('\t{}: {}'.format(key, value))
