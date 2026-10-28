import tkinter as tk

from gui.app import CodeGeneratorApp


def main():
    root = tk.Tk()
    CodeGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
