import json

from main import CHAR_TO_RAD_DIRECTORY, ENG_TO_CHARS_DIRECTORY


def json_to_dict(json_file: str) -> dict:
    """
    Load json file and return it as a dict
    :param json_file:
    :return:
    """
    with open(json_file) as f:
        data = json.load(f)
    f.close()
    return dict(data)


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


def load_eng_to_rads(kanji_to_rad_file, eng_to_kanji_file) -> dict[str, list[str]]:
    """
    Loads English words to radicals based on a Kanji to radical mapping, and English to Kanji mapping
    :return:
    """
    kanji_to_rads = json_to_dict(kanji_to_rad_file)
    eng_to_kanji = json_to_dict(eng_to_kanji_file)
    eng_to_rads = create_eng_to_rads(kanji_to_rads, eng_to_kanji)
    return eng_to_rads
