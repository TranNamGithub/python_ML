from tkinter import *
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog
import tkinter as tk
import dees

class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 20 # Interval in ms to get the latest frame
        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)
        # Update image on canvas
        self.update_image()
    def update_image(self):
        # Get the latest frame and convert image format
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to RGB
        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)
detect =True
root = tk.Tk()
root.geometry('1080x600')
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("Text files","*.txt*"),("all files","*.*")))
    filename ="r'"+filename+"'"
Button(root, text='browse file', command=browseFiles).grid(row =3,column =5)
MainWindow(root, dees.counting_vehicle(detect,r'E:\NAMHOC\python\highway.mp4', 'vehicles.avi'))
root.mainloop()