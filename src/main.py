import numpy as np

from config import HOME_PATH, config


class Runner():
    def __init__(self):
        pass

    @staticmethod
    def read_data(data_source):
        data_path = config['directories']['data_dir']
        data = np.load(f"{HOME_PATH}/{data_path}/{data_source}_data.npz")
        return data['f0'], data['f1']

    def run(self):
        # train_x, train_y = self.read_data(data_source='train')
        test_x, test_y = self.read_data(data_source='train')
        print(test_y)


if __name__ == '__main__':
    runner = Runner()
    runner.run()
