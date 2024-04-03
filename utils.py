import json


def json_to_dict(json_file: str) -> dict:
    """
    Load json file and return it as a dict
    :param json_file:
    :return:
    """
    return dict(json.load(open(json_file)))