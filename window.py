from tkinter import *

def window(): 
    window = Tk()
    window.title("sova_239")
    window.geometry("800x600")
    def display():
        out["text"] = main.get()
    main = Entry()
    out = Label()
    take = Button(text="Обработать", command=display)
    main.pack(anchor=S, pady=5)
    take.pack(anchor=S, pady=10)
    out.pack(anchor=S, pady=15)
    window.mainloop()
window()