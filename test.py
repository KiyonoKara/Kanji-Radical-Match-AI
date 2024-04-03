import unittest

from main import create_eng_to_rads

# example from https://docs.python.org/3/library/unittest.html
class TestStringMethods(unittest.TestCase):
    
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_eng_to_rads(self):
        # test the create_eng_to_rads function
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

        eng_to_rads = create_eng_to_rads(char_to_rad, eng_to_char)
        self.assertEqual(eng_to_rads["hello"], ["rad1", "rad2", "rad3", "rad4"])

if __name__ == '__main__':
    unittest.main()