import os
import configparser

HOME_PATH = os.environ.get('FLUENT_HOME')
config = configparser.ConfigParser()
config.read(os.path.join(HOME_PATH, 'env_config.ini'))