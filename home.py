import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLineEdit
import subprocess

class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Stock Market Predictor')
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.symbol_entry = QLineEdit(self)
        layout.addWidget(self.symbol_entry)

        self.lt_button = QPushButton('Long Term Prediction', self)
        self.lt_button.clicked.connect(self.run_long_term_prediction)
        layout.addWidget(self.lt_button)

        self.st_button = QPushButton('Short Term Prediction', self)
        self.st_button.clicked.connect(self.run_short_term_prediction)
        layout.addWidget(self.st_button)

        self.intraday_button = QPushButton('Intraday Prediction', self)
        self.intraday_button.clicked.connect(self.run_intraday_prediction)
        layout.addWidget(self.intraday_button)

        central_widget.setLayout(layout)

    def run_long_term_prediction(self):
        symbol = self.symbol_entry.text()
        subprocess.Popen(["python", "", symbol])

    def run_short_term_prediction(self):
        symbol = self.symbol_entry.text()
        subprocess.Popen(["python", "LT.py", symbol])

    def run_intraday_prediction(self):
        symbol = self.symbol_entry.text()
        subprocess.Popen(["python", "main.py", symbol])

def main():
    app = QApplication(sys.argv)
    ex = HomeWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
