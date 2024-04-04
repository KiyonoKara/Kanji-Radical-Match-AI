import unittest
import utils
import random


class TestSuite(unittest.TestCase):
    def test_eng_to_rads(self):
        # Test the create_eng_to_rads function
        eng_to_char = {
            "hello": ["h", "e", "l", "l", "o"],
            "world": ["w", "o", "r", "l", "d"]
        }

        char_to_rad = {
            "h": ["rad1"],
            "e": ["rad2"],
            "l": ["rad3"],
            "o": ["rad4"],
            "w": ["rad5"],
            "r": ["rad6"],
            "d": ["rad7"]
        }

        eng_to_rads = utils.create_eng_to_rads(char_to_rad, eng_to_char)
        self.assertEqual(eng_to_rads["hello"], ["rad1", "rad2", "rad3", "rad4"])

    def test_eng_to_rads2(self):
        # Test the create_eng_to_rads function
        eng_to_char = {
            "hello": ["h", "e", "l", "l", "o"],
            "world": ["w", "o", "r", "l", "d"]
        }

        char_to_rad = {
            "h": ["rad1", "rad8"],
            "e": ["rad2"],
            "l": ["rad3"],
            "o": ["rad4"],
            "w": ["rad5"],
            "r": ["rad6"],
            "d": ["rad7"]
        }

        eng_to_rads = utils.create_eng_to_rads(char_to_rad, eng_to_char)
        self.assertEqual(eng_to_rads["hello"], ["rad1", "rad8", "rad2", "rad3", "rad4"])

    def test_eng_to_rads3(self):
        # Test eng_to_rads function on actual data
        kanji_to_rad_dict = utils.json_to_dict("./data/kanji_to_radical.json")
        eng_to_kanji_dict = utils.json_to_dict("./data/english_to_kanji.json")
        e2k_keys = list(eng_to_kanji_dict.keys())
        e2k_sampled_keys = random.sample(e2k_keys, 10)
        e2k_sampled_dict = {key: eng_to_kanji_dict[key] for key in e2k_sampled_keys}

        eng_to_radicals = utils.create_eng_to_rads(kanji_to_rad_dict, e2k_sampled_dict)
        self.assertTrue(len(eng_to_radicals) == len(e2k_sampled_dict))


if __name__ == '__main__':
    unittest.main()
