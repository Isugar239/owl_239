from tkinter import *
window = Tk()
window.title("sova_239")
window.geometry("800x600")
main = Entry()
out = Label()
take = Button(text="Обработать", command=display)
def display():
    label["text"] = entry.get()
main.pack(anchor=S)
window.mainloop()