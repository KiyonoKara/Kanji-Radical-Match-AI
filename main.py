import json
import numpy as np
import torch
import utils
import pandas as pd
from sklearn import preprocessing

DATA_DIRECTORY_NAME = "data"
CHAR_TO_RAD_FILENAME = "kanji_to_radical.json"
CHAR_TO_RAD_DIRECTORY = f"{DATA_DIRECTORY_NAME}/{CHAR_TO_RAD_FILENAME}"
ENG_TO_CHARS_FILENAME = "english_to_kanji.json"
ENG_TO_CHARS_DIRECTORY = f"{DATA_DIRECTORY_NAME}/{ENG_TO_CHARS_FILENAME}"


# converted dictionary to pandas data frame so we can easily read the json
def reformat_data():
    eng_to_chars_data = pd.read_json(ENG_TO_CHARS_DIRECTORY)

    # add a new column to the data frame and switched column ordering
    eng_to_chars_data["English"] = eng_to_chars_data.index
    eng_to_chars_data.columns = ["Kanji", "English"]
    eng_to_chars_data = eng_to_chars_data[["English", "Kanji"]]

    # ensure the kanji column only contains one character, the english column will contain duplicate values
    if eng_to_chars_data["Kanji"].apply(len).max() > 1:
        eng_to_chars_data = eng_to_chars_data.explode("Kanji")
    return eng_to_chars_data


# encodes the data and labels from strings to randomly assigned numbers so they can be tensorized
# then convert data and labels to tensors
def preprocess_data(data):
    encoder = preprocessing.LabelEncoder()
    encoded_data = encoder.fit_transform(data["English"])
    encoded_label = encoder.fit_transform(data["Kanji"])
    data_tensor = torch.tensor(encoded_data)
    label_tensor = torch.tensor(encoded_label)
    return data_tensor, label_tensor


# model is a list of np vectors

def run_model(word, model):
    # throw word in model to see the radical vector
    radical_vect = []
    return radical_vect


def train_model(eng_to_rads):
    # use the english to radicals dictionary to train the model
    model = '''Add later'''
    # Dev note: Might recommend building a class-style torch feedforward model
    return model


def train_word(word_vec, true_lbl):
    # train a single word
    return


def save_model(model, filename):
    torch.save(model, filename)


def create_eng_to_rads(kanji_to_rads, eng_to_kanji) -> dict[str, list[str]]:
    """
    Use the kanji to radical dictionary and English to
    Character dictionary to construct the English to radical dictionary
    :param kanji_to_rads:
    :param eng_to_kanji:
    :return: dict of English words to radicals
    """

    eng_to_rads = dict()
    for eng_word in eng_to_kanji:
        # Create new dict entry for English word
        eng_to_rads[eng_word] = []
        for kanji in eng_to_kanji[eng_word]:
            # Add unique radicals to English word entry
            if kanji in kanji_to_rads:
                for rad in kanji_to_rads[kanji]:
                    if rad not in eng_to_rads[eng_word]:
                        eng_to_rads[eng_word].append(rad)
    return eng_to_rads


def load_eng_to_rads():
    char_to_rads = utils.json_to_dict(CHAR_TO_RAD_DIRECTORY)
    eng_to_chars = utils.json_to_dict(ENG_TO_CHARS_DIRECTORY)
    eng_to_rads = create_eng_to_rads(char_to_rads, eng_to_chars)
    return eng_to_rads


def main():
    return
    # put in whatever you want to here for debugging locally


if __name__ == "__main__":
    main()
