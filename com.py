import os; os.system('cls')
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
        arduino.write(bytes('s', 'utf-8'))  # Command to start conveyor
        conveyor_button.config(text="Stop Conveyor", bg="#FF6666")
        print("Conveyor started")
    conveyor_on = not conveyor_on

# GUI setup
win = tk.Tk()
win.title('Selector')
win.minsize(300, 100)

label = tk.Label(win, text=' Gusset State ', font='calibri', fg='blue')
label.grid(column=1, row=1)

btn_bad = tk.Button(win, text='Defective', font='calibri 12 bold', bg='#FF6666', command=bad)
btn_bad.grid(column=1, row=2)

btn_good = tk.Button(win, text='Non-defective', font='calibri 12 bold', bg='#90EE90', command=good)
btn_good.grid(column=2, row=2)

conveyor_button = tk.Button(win, text="Start Conveyor", font='calibri 12 bold', bg='#90EE90', command=toggle_conveyor)
conveyor_button.grid(column=1, row=3, columnspan=2)

win.mainloop()
