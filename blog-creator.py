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

    print("Generating blog card...")

    with open("./templates/blog-card.html", "r", encoding="UTF-8") as f:
        blog_card_html = f.read()

    blog_card_html = blog_card_html.replace("{{ title }}", original_name)
    blog_card_html = blog_card_html.replace("{{ date }}", f"{day}/{month}/{year}")
    blog_card_html = blog_card_html.replace(
        "{{ description }}",
        "PyTorch ile MNIST veri kümesi üzerinde temellere dönüş yaparak, bir sinir ağı modeli oluşturacağız. Bu modeli eğitip, test ederek, modelin doğruluğunu ölçeceğiz.",
    )
    blog_card_html = blog_card_html.replace(
        "{{ route_path }}", directory_path.replace("\\", "/")[1:]
    )

    print("-" * 50)
    print(blog_card_html)


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

    html = markdown2.markdown(
        content,
        extras=[
            "fenced-code-blocks",
            "cuddled-lists",
            "target-blank-links",
            "task_list",
        ],
    )

    return html


base_path = "./blog"
name = "Temellere Dönüş, PyTorch ile MNIST"
blog_path = "./blogs/temellere_donus_py_torch_ile_mnist.md"
blog_html = convert_markdown_to_html(blog_path)
create_directories(base_path, name, blog_html)
