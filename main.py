import json

CHAR_TO_RAD_FILENAME = "char_to_rad.json"
ENG_TO_CHARS_FILENAME = "eng_to_chars.json"



def load_files():
    char_to_rads = load_dict(CHAR_TO_RAD_FILENAME)
    eng_to_chars = load_dict(ENG_TO_CHARS_FILENAME)
    eng_to_rads = create_eng_to_rads(char_to_rads, eng_to_chars)
    
def load_dict(filename):
    # Check if we need to modify the open method because of utf stuff!
    with open(filename) as file:
        dict = json.load(file)
    return dict

def main():
    load_files()



if __name__ == "__main__":
    main()