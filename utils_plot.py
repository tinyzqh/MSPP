import pandas as pd
import os
import math
import matplotlib.pyplot as plt

class plot_txts(object):
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

def main():
    instantiation = plot_txts()
    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-swingup_seed_1_dreamer_kl_beta_0_action_repeat_8")
    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-balance_1_dreamer")

    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-balance_global_split_test_seed_1_p2p_kl_beta_2_action_repeat_8")


    # standard: obs_loss 2; reward_loss 0.02; kl loss 3.2
    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-balance/cartpole-balance_seed_1_dreamer_kl_beta_0_action_repeat_8")

    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-balance_global_split_test_seed_1_dreamer_kl_beta_1_action_repeat_8")
    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-balance_global_split_test_seed_1_dreamer_kl_beta_1_action_repeat_2")

    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-balance/cartpole-balance_global_split_test_seed_1_dreamer_kl_beta_0_action_repeat_8")
    all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/Master's_thesis/dreamer/results/cartpole-balance_final_seed_1_dreamer_kl_beta_1_action_repeat_2_explore")
    instantiation.process_txts(all_txt_path)

if __name__ == "__main__":
    main()
