
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
root = Tk()

def browsefunc():
    filename =askopenfilename()
    pathlabel.config(text=filename)

browsebutton = Button(root, text="Upload", command=browsefunc)
browsebutton.pack()

pathlabel = Label(root)
pathlabel.pack()
root.mainloop()