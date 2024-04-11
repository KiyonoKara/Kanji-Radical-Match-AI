import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import utils as u
from utils import KanjiFFNN

TOP_TAKE = 10
eng_to_rads = dict(list(u.load_eng_to_rads().items()))
eng_tens, rad_tens, eng_vocab, rad_vocab = u.dict_to_tensors(eng_to_rads)
e2r_model = KanjiFFNN(eng_tens.size(1), rad_tens.size(1))
QUIT_MSG = '!quit'


def load_e2r_model():
    e2r_model.load_state_dict(torch.load("./models/model_state_dict.pt"))


def radical_distribution_generator(input_word):
    input_tensor = u.get_tensor_from_word(input_word, eng_tens, eng_vocab)

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
