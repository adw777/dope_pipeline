import os
import yaml
import logging
import logging.config

def setupLogging(
    defaultPath='logging.yaml',
    defaultLevel=logging.INFO,
    envKey='LOG_CFG'
):
    """
    Setup logging configuration
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    path = os.path.join(script_dir, defaultPath)
    value = os.getenv(envKey, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            
        for handler in config['handlers'].values():
            if 'filename' in handler:
                log_path = os.path.join(script_dir, handler['filename'])
                handler['filename'] = log_path
                
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=defaultLevel)