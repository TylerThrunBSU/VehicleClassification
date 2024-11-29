import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# Paths to labels and images
labels_path = r'C:\Users\tthrun\Desktop\AI Coding Section\Yolo Test\Copy_Of_Stanford_Cars\train\labels'
images_path = r'C:\Users\tthrun\Desktop\AI Coding Section\Yolo Test\Copy_Of_Stanford_Cars\train\images'

# Define vehicle types
vehicle_types = [
    "Pedestrian",
    "Pedestrian Cyclist",
    "Motorcycle",
    "Sedan",
    "SUV",
    "Passenger Truck",
    "Single Unit Truck",
    "Semi-Truck",
    "Illegible (Throw out)",
    "Not a Vehicle"
]

class VehicleClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Vehicle Classifier")
        
        # Load image files
        self.image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.current_image_index = 0
        
        # Create image label
        self.image_label = tk.Label(master)
        self.image_label.pack()
        
        # Create classification buttons
        self.classification_buttons = []
        for i, vehicle_type in enumerate(vehicle_types):
            button = tk.Button(master, text=f"{i}: {vehicle_type}", command=lambda c=i: self.classify_image(c))
            button.pack(side=tk.LEFT)
            self.classification_buttons.append(button)

        self.show_image()
        
    def show_image(self):
        if self.current_image_index < len(self.image_files):
            image_name = self.image_files[self.current_image_index]
            image_path = os.path.join(images_path, image_name)
            img = Image.open(image_path)
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
            
    def classify_image(self, vehicle_type):
        image_name = self.image_files[self.current_image_index]
        update_yolo_label(image_name, vehicle_type)
        print(f"Label updated for {image_name} with vehicle type '{vehicle_types[vehicle_type]}'.")
        
        self.current_image_index += 1
        if self.current_image_index < len(self.image_files):
            self.show_image()
        else:
            messagebox.showinfo("Info", "All images have been classified.")
            self.master.quit()

def update_yolo_label(image_name, vehicle_type):
    label_name = os.path.splitext(image_name)[0] + '.txt'
    label_path = os.path.join(labels_path, label_name)

    if not os.path.exists(label_path):
        print(f"Label file for {image_name} does not exist.")
        return

    with open(label_path, 'r') as file:
        lines = file.readlines()

    if lines:
        lines[0] = f"{vehicle_type} " + ' '.join(lines[0].split()[1:])

    with open(label_path, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleClassifierApp(root)
    root.mainloop()