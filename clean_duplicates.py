#! /usr/bin/env python

import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
import sys
import signal
from PIL import Image, ImageTk

adjusted_base = '/mnt/NAS/Photos'

titles = ['not_added', 'in_database']
image_pairs = []
with open('dd.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        duplicate = row[titles[0]].replace('/photos', adjusted_base)
        populated = row[titles[1]].replace('/photos', adjusted_base)
        assert os.path.exists(duplicate), f'{duplicate} doesn\'t exist!'
        assert os.path.exists(populated)
        image_pairs.append( (duplicate, populated) )

class ImageViewer(tk.Tk):
    def __init__(self, image_pairs, max_width=800, max_height=800):
        super().__init__()

        self.image_pairs = image_pairs
        self.index = 0
        self.max_width = max_width
        self.max_height = max_height

        self.title("Image Viewer")

        # Load the first pair of images
        self.img1 = self.load_image(self.image_pairs[self.index][0])
        self.img2 = self.load_image(self.image_pairs[self.index][1])
        
        # Create labels to display the images
        self.label1 = tk.Label(self, image=self.img1)
        self.label2 = tk.Label(self, image=self.img2)

        # Arrange the labels in the window
        self.label1.grid(row=0, column=0)
        self.label2.grid(row=0, column=1)

        # Create Yes and No buttons
        self.yes_button = tk.Button(self, text="Is duplicate", command=self.on_yes_click)
        self.no_button = tk.Button(self, text="Not duplicate", command=self.on_no_click)

        # Arrange the buttons in the window
        self.yes_button.grid(row=1, column=0)
        self.no_button.grid(row=1, column=1)

        # Bind the close event to a custom method
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Set up the signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.signal_handler)

    def load_image(self, image_path):
        # Open the image
        img = Image.open(image_path)
        # Resize the image while maintaining aspect ratio
        img.thumbnail((self.max_width, self.max_height))
        return ImageTk.PhotoImage(img)

    def update_images(self):
        if self.index < len(self.image_pairs):
            self.img1 = self.load_image(self.image_pairs[self.index][0])
            self.img2 = self.load_image(self.image_pairs[self.index][1])
            
            self.label1.config(image=self.img1)
            self.label2.config(image=self.img2)
        else:
            self.yes_button.config(state=tk.DISABLED)
            self.no_button.config(state=tk.DISABLED)
            print("No more images to display.")

    def on_yes_click(self):
        # Delete the image
        print(self.image_pairs[self.index][0])
        print("Not deleting: ", self.image_pairs[self.index][1])
        os.remove(self.image_pairs[self.index][0])
        self.image_pairs.pop(0)
        print(len(self.image_pairs))

        self.index += 1
        self.update_images()

    def on_no_click(self):

        self.index += 1
        self.update_images()
    
    def on_close(self):
        print("Window is closing.")
        # Write the image pairs back to CSV.

        with open('dd.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=titles)
            writer.writeheader()
            for row in self.image_pairs:
                print(row)
                duplicate = row[0].replace(adjusted_base, '/photos')
                populated = row[1].replace(adjusted_base, '/photos')
                aa = {titles[0]: duplicate, titles[1]: populated}
                writer.writerow(aa)
        self.destroy()

    def signal_handler(self, sig, frame):
        print("Caught Ctrl-C (SIGINT).")
        self.on_close()
        sys.exit(0)

# Create the application
app = ImageViewer(image_pairs)
app.mainloop()

