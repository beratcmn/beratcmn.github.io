import os
import re
from datetime import datetime
import markdown2


def create_directories(base_path, name, raw_blog_html=""):
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    original_name = name
    name = to_snake_case(name)

    directory_path = os.path.join(base_path, year, month, day, name)

    os.makedirs(directory_path, exist_ok=True)

    print(f"Directories created: {directory_path}")

    with open("./templates/blog.html", "r", encoding="UTF-8") as f:
        blog_html = f.read()

    blog_html = blog_html.replace("{{ content }}", raw_blog_html)
    blog_html = blog_html.replace("{{ title }}", original_name)

    file_path = os.path.join(directory_path, "index.html")
    with open(file_path, "w", encoding="UTF-8") as f:
        f.write(blog_html)

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


def convert_markdown_to_html(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        content = f.read()

    html = markdown2.markdown(content, extras=["fenced-code-blocks"])

    return html


base_path = "./blog"
name = "Temellere Dönüş, PyTorch ile MNIST"
blog_path = "./blogs/temellere_donus_py_torch_ile_mnist.md"
blog_html = convert_markdown_to_html(blog_path)
create_directories(base_path, name, blog_html)
