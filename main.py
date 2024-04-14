import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils as u
from utils import KanjiFFNN


class KanjiFFNN_V1(nn.Module):
    def __init__(self, eng_vocab_size: int, radical_vocab_size: int):
        super(KanjiFFNN_V1, self).__init__()
        # Hidden layer
        self.input = nn.Linear(eng_vocab_size, 600)
        self.hid1 = nn.Linear(600, 400)
        self.hid2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, radical_vocab_size)

    def forward(self, x):
        """
        Forward propagation of the model
        :param x: Data
        :return:
        """
        x = F.relu(self.input(x))
        x = F.relu(self.hid1(x))
        x = F.relu(self.hid2(x))
        x = F.sigmoid(self.output(x))
        return x

    def train_fit(self,
                  eng_tensors: torch.Tensor,
                  rad_tensors: torch.Tensor,
                  optimizer: optim.Optimizer,
                  criterion=nn.MSELoss(),
                  epochs=100,
                  scheduler: optim.lr_scheduler.LRScheduler = None,
                  verbose=False):
        """
        Forwards itself to train_model function
        :param eng_tensors:
        :param rad_tensors:
        :param optimizer:
        :param criterion:
        :param epochs:
        :param scheduler:
        :param verbose:
        :return:
        """
        return u.train_model(self, eng_tensors, rad_tensors, optimizer, criterion, epochs, scheduler, verbose)


TOP_TAKE = 10
eng_to_rads = dict(list(u.load_eng_to_rads().items()))
eng_tens, rad_tens, eng_vocab, rad_vocab = u.dict_to_tensors(eng_to_rads)
e2r_model = KanjiFFNN_V1(eng_tens.size(1), rad_tens.size(1))
QUIT_MSG = '!quit'


def load_e2r_model():
    e2r_model.load_state_dict(torch.load("./models/model_state_dict.pt"))


def radical_distribution_generator(input_word):
    input_tensor = u.get_tensor_from_word(input_word, eng_tens, eng_vocab, verbose=True)

    pred_tensor = e2r_model(input_tensor)

    output_probs = pred_tensor.detach().numpy().squeeze()
    radical_probs = [(radical, prob) for radical, prob in zip(rad_vocab, output_probs)]
    sorted_radical_probs = sorted(radical_probs, key=lambda x: x[1], reverse=True)

    radicals, probabilities = zip(*sorted_radical_probs)

    radicals_top = list(radicals[:TOP_TAKE])
    probabilities_top = list(probabilities[:TOP_TAKE])

    fig, axs = plt.subplots()
    axs.bar(range(TOP_TAKE), probabilities_top)
    fprop = fm.FontProperties(fname='NotoSansCJKtc-Regular.otf')
    axs.set_xticks(range(TOP_TAKE), radicals_top, fontproperties=fprop)
    axs.set_xlabel('Radicals')
    axs.set_ylabel('Probabilities')
    axs.set_title(f'Top {TOP_TAKE} Radicals Most Likely Associated With \"{input_word}\"')
    plt.ylim(0.0, 1.0)

    plt.show()


def main():
    load_e2r_model()

    print('English to Radical Matching\n'
          '*.+:.☆　英単語→(日中)部首　☆.:+.*\n'
          f'(Type \'{QUIT_MSG}\' to quit anytime)\n')

    while True:
        input_text = input("Enter your input word or phrase: ").strip()
        if input_text == QUIT_MSG:
            exit()

        radical_distribution_generator(input_text.strip())


if __name__ == "__main__":
    main()
