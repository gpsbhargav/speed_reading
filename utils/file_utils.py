import os
import glob
import pickle


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    # else:
    #     assert overwrite == True, "Experiment directory already exists."


def pickler(pkl_file, obj):
    with open(pkl_file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickler(pkl_file):
    with open(pkl_file, "rb") as f:
        obj = pickle.load(f)
    return obj


class Logger:
    def __init__(self, file_path, print_to_screen=True):
        self.file_path = file_path
        self.print_to_screen = print_to_screen
        with open(file_path, "w") as log_file_handle:
            log_file_handle.write(
                "================================================= \n"
            )

    def write_log(self, log_str):
        with open(self.file_path, "a") as log_file_handle:
            log_file_handle.write(log_str)
            log_file_handle.write("\n")
        if self.print_to_screen:
            print(log_str)
