import os
import sys

from numpy import negative
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QPlainTextEdit, QPushButton,
                             QVBoxLayout, QWidget)

from detect import predict


class ImageLoader(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Loader")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.button = QPushButton("input image", self)
        self.button.clicked.connect(self.load_image)
        self.layout.addWidget(self.button)

        self.button_generate = QPushButton("generate", self)
        self.button_generate.clicked.connect(self.generate)
        self.layout.addWidget(self.button_generate)

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        # Adding an input textbox
        self.textbox2 = QLineEdit(self)
        self.textbox2.setPlaceholderText("negative prompt")  # Placeholder text
        self.layout.addWidget(self.textbox2)

        # Output textbox
        self.textbox1 = QPlainTextEdit(self)
        self.textbox1.setReadOnly(True)
        self.layout.addWidget(self.textbox1)

    def load_image(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "./")

        if fname[0]:
            image = Image.open(fname[0])
            image = image.resize((250, 250), Image.LANCZOS)
            image.save("./target/temp.png")
            pixmap = QPixmap("./target/temp.png")
            self.label.setPixmap(pixmap)

    def generate(self):
        # Get the text from the textbox2
        labels = []
        if os.path.exists("./target/temp.png"):
            labels = predict("./target/temp.png")
            print(labels)
        negative_prompt = self.textbox2.text()
        if len(labels) != 0:
            text = "\n".join(
                ask_recipe(labels, list(negative_prompt.split(" ")))
            )  # ["beef", "pork", "chicken"]
        else:
            text = "Let's go to super market!"
        self.textbox1.setPlainText(text)


from chatgpt import chat

prompt = "I require someone who can suggest delicious recipes that includes foods which are nutritionally beneficial but also easy & not time consuming enough therefore suitable for busy people like us among other factors such as cost effectiveness so overall dish ends up being healthy yet economical at same time! My first request – \
    “Under the condition that a sweet potato is considered a potato describe a simple dish like curry using the following ingredients and how to prepare it. \n \
    !ingredients! \n However, please do not use the following ingredients and cookware. \n !negative!"
counter = 0


def ask_recipe(ingredients_list: list, negative_list: list):
    # chatgptで生成した文を生成
    ingredients = ""
    for ingredient in ingredients_list:
        ingredients += ingredient + "\n"
    negatives = ""
    for negative in negative_list:
        negatives += negative + "\n"
    new_prompt = prompt.replace("!ingredients!", ingredients).replace(
        "!negative!", negatives
    )
    generated_sentences = list(
        chat(new_prompt)["choices"][0]["message"]["content"].split("\n")
    )
    return generated_sentences


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageLoader()
    ex.show()
    sys.exit(app.exec_())
