import os
import re
from datetime import datetime


def create_directories(base_path, name):
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    directory_path = os.path.join(base_path, year, month, day, name)

    os.makedirs(directory_path, exist_ok=True)

    print(f"Directories created: {directory_path}")

    # Create a HTML file
    file_path = os.path.join(directory_path, "index.html")
    with open(file_path, "w") as f:
        f.write(
            f"<!DOCTYPE html>\n<html>\n<head>\n<title>{name}</title>\n</head>\n<body>\n<h1>{name}</h1>\n</body>\n</html>"
        )

    print(f"File created: {file_path}")


def to_snake_case(s):
    def replace_turkish_letters(s):
        replacements = {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
            "Ç": "C",
            "Ğ": "G",
            "İ": "I",
            "Ö": "O",
            "Ş": "S",
            "Ü": "U",
        }

        for turkish, english in replacements.items():
            s = s.replace(turkish, english)

        return s

    s = replace_turkish_letters(s)
    s = re.sub(r"[\s\W]+", "_", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    return s.lower()


base_path = "./blog"
name = "Temellere Dönüş, PyTorch ile MNIST"
create_directories(base_path, to_snake_case(name))
