import os
import configparser

HOME_PATH = os.environ.get('FLUENT_HOME')
CONFIG_PATH = os.path.join(HOME_PATH, 'config/env_config.ini')
config = configparser.ConfigParser()
config.read(CONFIG_PATH)