import json
import numpy as np
import torch
import utils
import pandas as pd
from sklearn import preprocessing


def dict_to_tensors(dict):
    """
    Converts the dict of English words to radicals into tensors that can be used by the network
    :return:
    """
    # creates pandas dataframe of dict to convert it to a format that can be made into a tensor
    dataframe = pd.DataFrame(dict.items(), columns=["English", "Radical"])
    dataframe = dataframe.explode("Radical")

    # encodes and creates tensors of the input and output
    encoder = preprocessing.LabelEncoder()
    encoded_eng = encoder.fit_transform(dataframe["English"])
    encoded_rad = encoder.fit_transform(dataframe["Radical"])
    eng_tensor = torch.tensor(encoded_eng)
    rad_tensor = torch.tensor(encoded_rad)
    return eng_tensor, rad_tensor


# model is a list of np vectors
def predict(word, model):
    """
    [ADD DOCUMENTATION HERE]
    [INCLUDE SIGNATURE ON FUNCTION]
    :param word:
    :param model:
    :return:
    """
    # throw word in model to see the radical vector
    radical_vect = []
    return radical_vect


def train_model(eng_tensor: torch.Tensor, rad_tensor: torch.Tensor):
    # use the english to radicals dictionary to train the model
    model = '''Add later'''
    # Dev note: Might recommend building a class-style torch feedforward model
    return model


def train_word(word_vec, true_lbl):
    # train a single word
    return


def save_model(model, filename):
    torch.save(model, filename)


def main():
    return
    # put in whatever you want to here for debugging locally


if __name__ == "__main__":
    main()
