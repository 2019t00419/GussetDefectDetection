import os
import tkinter as tk
from tkinter import ttk, messagebox

def submit_data():
    messagebox.showinfo("Submitted Data", "Data:")

def main():
    window = tk.Tk()
    window.geometry("1300x1000")
    window.title("Gusset Quality Control")

    # Apply a custom dark theme
    style = ttk.Style(window)
    window.tk_setPalette(background='#2E2E2E', foreground='#FFFFFF', activeBackground='#3E3E3E', activeForeground='#FFFFFF')

    style.configure('TLabel', background='#2E2E2E', foreground='#FFFFFF')
    style.configure('TEntry', fieldbackground='#3E3E3E', foreground='#FFFFFF')
    style.configure('TButton', background='#3E3E3E', foreground='#FFFFFF')

    # Create widgets with ttk
    label = ttk.Label(window, text="Enter Quality Control Data:")
    global entry
    entry = ttk.Entry(window)
    button = ttk.Button(window, text="Submit", command=submit_data)

    # Layout widgets
    label.pack(pady=10)
    entry.pack(pady=10)
    button.pack(pady=10)

    window.mainloop()

if __name__ == '__main__':
    main()
