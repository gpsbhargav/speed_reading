import os
import glob
import pickle
import json


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


class JSONLLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.create_log_file()

    def create_log_file(self):
        with open(self.log_file, "w") as out_file:
            pass

    def write_log(self, dict_in):
        with open(self.log_file, "a") as out_file:
            json_str = json.dumps(dict_in)
            out_file.write(json_str)
            out_file.write("\n")
