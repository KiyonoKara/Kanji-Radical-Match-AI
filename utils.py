import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn import preprocessing

DATA_DIRECTORY_NAME = "data"
CHAR_TO_RAD_FILENAME = "kanji_to_radical.json"
CHAR_TO_RAD_DIRECTORY = f"{DATA_DIRECTORY_NAME}/{CHAR_TO_RAD_FILENAME}"
ENG_TO_CHARS_FILENAME = "english_to_kanji.json"
ENG_TO_CHARS_DIRECTORY = f"{DATA_DIRECTORY_NAME}/{ENG_TO_CHARS_FILENAME}"


def json_to_dict(json_file: str) -> dict:
    """
    Load json file and return it as a dict
    :param json_file:
    :return:
    """
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    return dict(data)

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


def load_eng_to_rads() -> dict[str, list[str]]:
    """
    Loads English words to radicals based on a Kanji to radical mapping, and English to Kanji mapping
    :return:
    """
    kanji_to_rads = json_to_dict(CHAR_TO_RAD_DIRECTORY)
    eng_to_kanji = json_to_dict(ENG_TO_CHARS_DIRECTORY)
    eng_to_rads = create_eng_to_rads(kanji_to_rads, eng_to_kanji)
    return eng_to_rads


def train_model(model: nn.Module,
                eng_tensor: torch.Tensor,
                rad_tensor: torch.Tensor,
                optimizer: optim.Optimizer,
                criterion=nn.MSELoss(),
                epochs=100,
                verbose=False):
    """
    Trains the model based on all of its information and parameters
    :param model:
    :param eng_tensor:
    :param rad_tensor:
    :param optimizer:
    :param criterion:
    :param epochs:
    :param verbose: Whether to print the loss during training
    :return:
    """

    for i in range(0, epochs):
        for eng, rad in zip(eng_tensor, rad_tensor):
            # Zero the gradient buffers
            optimizer.zero_grad()
            output = model(eng)
            # Large
            loss = criterion(rad, output)
            loss.backward()
            # Update
            optimizer.step()
            if verbose:
                print("Epoch {: >8} Loss: {}".format(i, loss.data.numpy()))


class KanjiFFNN(nn.Module):
    def __init__(self, eng_vocab_size: int, radical_vocab_size: int, nodes: int):
        super(KanjiFFNN, self).__init__()
        # Hidden layer
        self.hid1 = nn.Linear(eng_vocab_size, nodes)
        # Output layer
        self.out1 = nn.Linear(nodes, radical_vocab_size)

    def forward(self, x):
        """
        Forward propagation of the model
        :param x: Data
        :return:
        """
        # Pass input x to hidden layer
        x = self.hid1(x)
        # Apply ReLU activation function to output of first layer
        x = F.relu(x)
        # Pass the output from the previous layer to the output layer
        x = self.out1(x)
        # Apply sigmoid to output
        x = F.sigmoid(x)
        return x
