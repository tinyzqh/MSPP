import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class plot_single_txts(object):
    def __init__(self):
        pass

    def get_all_txt_filename(self, root_path):
        for root, dir, files in os.walk(root_path):
            all_txt_path = [os.path.join(root, i) for i in files if i.endswith(".txt") and i!="test_rewards_steps_step.txt"]
            return all_txt_path
    def process_txts(self, txt_paths):
        assert isinstance(txt_paths, list)
        row = math.ceil(math.sqrt(len(txt_paths)))
        column = math.ceil(len(txt_paths)/row)
        for i, txt_path in enumerate(txt_paths):
            plt.subplot(row, column, i+1)
            self.process_txt(txt_path)
        plt.show()
    def process_txt(self, file_name):
        title = file_name.split("/")[-1][:-4]
        x_label = title.split("_")[-1]
        y_label = title.split("_")[-2]

        print("filename {}".format(file_name))
        df = pd.read_csv(file_name, sep=" ")
        columns_name = df.columns
        x_data = columns_name[0]
        y_datas = columns_name[1:]
        for index, y_data in enumerate(y_datas):
            plt.plot(df[x_data].values, df[y_data].values, label=y_datas[index])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            # plt.tight_layout()

class plot_multi_txt(object):
    def __init__(self,root_path):
        self.root_path = root_path
        self.all_dir_path = None
        self.filepaths = []
        self.file_name = "train_rewards_episode.txt"
        
    def get_files_path(self):
        for root, dir, files in os.walk(self.root_path):
            all_algorithms_dir = [os.path.join(root,i) for i in dir]
            return all_algorithms_dir

    def plot_all_algorithms_data(self):
        sns.set(style="darkgrid", font_scale=1.5, font='serif', rc={'figure.figsize': (10, 8)})
        all_algorithms_dir = self.get_files_path()
        for dir in all_algorithms_dir:
            self.all_files_path(rootDir=dir) # get all txt files. -> saved in self.filepaths
            x_data, y_data = self.get_data(self.filepaths)
            if dir.split('/')[-1] == 'dreamer':
                sns.tsplot(time=x_data, data=y_data, color='r', condition='action scale 1', err_style='ci_band', estimator=np.median)
            if dir.split('/')[-1] == 'two_scale':
                sns.tsplot(time=x_data, data=y_data, color='b', condition='action scale 2', err_style='ci_band', estimator=np.median)
            self.filepaths = [] # clear -> self.filepaths

        plt.ylabel("Return")
        plt.xlabel("Environment Steps(x100)")
        plt.title(self.root_path.split('/')[-1])
        plt.xlim(-0.5, 900)
        plt.ylim(-0.5, 1000)
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # plt.savefig('/home/hzq/Figure_1.png')
        plt.show()

    def all_files_path(self, rootDir):
        for root, dirs, files in os.walk(rootDir):
            for file in files:
                file_path = os.path.join(root, file)
                if file == self.file_name and (file_path not in self.filepaths):
                    self.filepaths.append(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                self.all_files_path(dir_path)

    def smooth(self, data, sm=200):
        smooth_data = []
        for d in data:
            y = np.ones(sm) * 1 / sm
            d = np.array(d).flatten()
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
        return smooth_data

    def get_data(self, files):
        x_out_data = []
        y_out_data = []
        for file_name in files:
            print("filename {}".format(file_name))
            df = pd.read_csv(file_name, sep=" ")
            columns_name = df.columns
            x_data = columns_name[0]
            y_datas = columns_name[1:]
            if x_out_data.__len__() == 0: x_out_data.append(df[x_data].values)
            y_out_data.append(df[y_datas].values)
        return x_out_data, self.smooth(y_out_data)

def main():
    # plot_multi_txt(root_path="/home/hzq/final-datas/cartpole-balance").plot_all_algorithms_data()


    instantiation = plot_single_txts()
    all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master_thesis/Kagebunsin-no-jyutu/results/cartpole-balance_final_seed_5_p2p_kl_beta_1_action_scale_2_no_explore")
    instantiation.process_txts(all_txt_path)

if __name__ == "__main__":
    main()
