import json
import numpy as np
import torch

DATA_DIRECTORY_NAME = "data"
CHAR_TO_RAD_FILENAME = "char_to_rad.json"
CHAR_TO_RAD_DIRECTORY = f"{DATA_DIRECTORY_NAME}/{CHAR_TO_RAD_FILENAME}"
ENG_TO_CHARS_FILENAME = "eng_to_chars.json"
ENG_TO_CHARS_DIRECTORY = f"{DATA_DIRECTORY_NAME}/{ENG_TO_CHARS_FILENAME}"

# model is a list of np vectors

def run_model(word, model):
    # throw word in model to see the radical vector
    radical_vect = []
    return radical_vect

def train_model(eng_to_rads):
    # use the english to radicals dictionary to train the model
    model = []
    return model

def train_word(word_vec, true_lbl):
    # train a single word
    return

def save_model(model, filename):
    torch.save()

def create_eng_to_rads(char_to_rads, eng_to_chars):
    eng_to_rads = dict()

    # TODO: use the character to radical dictionary and english to
    # character dictionary to construct the english to radical dictionary

    for eng_word in eng_to_chars:

        # create new dict entry for eng word
        eng_to_rads[eng_word] = []
        for char in eng_to_chars[eng_word]:

            # add unique radicals to eng word entry
            if char in char_to_rads:
                for rad in char_to_rads[char]:
                    if (rad not in eng_to_rads[eng_word]):
                        eng_to_rads[eng_word].append(rad)
    return eng_to_rads

def load_json_dict(filename):
    # Check if we need to modify the open method because of utf stuff!
    with open(filename) as file:
        json_dict = json.load(file)
    return json_dict

def load_eng_to_rads():
    char_to_rads = load_json_dict(CHAR_TO_RAD_DIRECTORY)
    eng_to_chars = load_json_dict(ENG_TO_CHARS_DIRECTORY)
    eng_to_rads = create_eng_to_rads(char_to_rads, eng_to_chars)
    return eng_to_rads

def main():
    return
    # put in whatever you want to here for debugging locally

if __name__ == "__main__":
    main()