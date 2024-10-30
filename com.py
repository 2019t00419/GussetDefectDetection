import os
import tkinter as tk
import serial

# Serial communication setup
arduino = serial.Serial(port='COM3', baudrate=115200, timeout=0.1)

# Track conveyor state
conveyor_on = False

def bad():
    arduino.write(bytes('b', 'utf-8'))
    print("Defective - moving backward")

def good():
    arduino.write(bytes('g', 'utf-8'))
    print("Non-defective - moving forward")

def toggle_conveyor():
    global conveyor_on
    if conveyor_on:
        arduino.write(bytes('c', 'utf-8'))  # Command to stop conveyor
        conveyor_button.config(text="Start Conveyor", bg="#90EE90")
        print("Conveyor stopped")
    else:
        if reverse_var.get():  # Check if the reverse checkbox is checked
            arduino.write(bytes('r', 'utf-8'))  # Command to start conveyor in reverse
            print("Conveyor started in reverse")
        else:
            arduino.write(bytes('s', 'utf-8'))  # Command to start conveyor forward
            print("Conveyor started forward")
        conveyor_button.config(text="Stop Conveyor", bg="#FF6666")
    conveyor_on = not conveyor_on

# Function to map keys to button actions
def key_press(event):
    if event.keysym == 'Down':  # Map Down arrow
        bad()
    elif event.keysym == 'Up':  # Map Up arrow
        good()
    elif event.keysym == 'Return':
        toggle_conveyor()

# GUI setup
win = tk.Tk()
win.title('Selector')
win.minsize(300, 150)

label = tk.Label(win, text=' Gusset State ', font='calibri', fg='blue')
label.grid(column=1, row=1)

btn_bad = tk.Button(win, text='Defective', font='calibri 12 bold', bg='#FF6666', command=bad)
btn_bad.grid(column=1, row=2)

btn_good = tk.Button(win, text='Non-defective', font='calibri 12 bold', bg='#90EE90', command=good)
btn_good.grid(column=2, row=2)

conveyor_button = tk.Button(win, text="Start Conveyor", font='calibri 12 bold', bg='#90EE90', command=toggle_conveyor)
conveyor_button.grid(column=1, row=3, columnspan=2)

# Checkbox to set conveyor direction
reverse_var = tk.BooleanVar()
reverse_checkbox = tk.Checkbutton(win, text="Reverse Direction", variable=reverse_var, font='calibri 10')
reverse_checkbox.grid(column=1, row=4, columnspan=2)

# Bind keyboard keys to functions
win.bind('<Key>', key_press)

win.mainloop()
