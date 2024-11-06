import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from tkinter import filedialog
from tkinter import *
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((254, 254)),
    transforms.ToTensor(),  # Converting the image into PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# CNN Model Architecture
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1) # First Conventional Layer
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1) # Second Conventional Layer
        self.pool = torch.nn.MaxPool2d(2, 2) # Max Pooling
        self.fc1 = torch.nn.Linear(32 * 62 * 62, 128) # Fulling Connected Dense Layer
        self.fc2 = torch.nn.Linear(128, num_classes) # Second fulling connected layer - outputs final classification score

    # Method that describe how each image input is passed through each layers during forward propagation process
    def forward(self, x):
        x = F.relu(self.conv1(x)) # Applies C. layer followed by ReLu activation function
        x = self.pool(x) # Applies pooling layer
        x = F.relu(self.conv2(x)) # Second C. layer
        x = self.pool(x) # Applies pooling layer
        x = x.view(-1, 32 * 62 * 62) # Flatten tensor into a 1D vector
        x = F.relu(self.fc1(x)) # passes the flatten tensor through the first C.layer followed by ReLu
        x = self.fc2(x) # Pass the output to second fully connected layer
        return F.log_softmax(x, dim=1) # applies softmax function to output tensor

num_classes = 54
model = CNN(num_classes=num_classes)
model.load_state_dict(torch.load('uno_model.pt'))
model.eval()

class_names = ['blue_0', 'blue_1', 'blue_2', 'blue_3', 'blue_4', 'blue_5', 'blue_6', 'blue_7', 'blue_8', 'blue_9', 'blue_draw_two', 'blue_reverse', 'blue_skip',
               'green_0', 'green_1', 'green_2', 'green_3', 'green_4', 'green_5', 'green_6', 'green_7', 'green_8', 'green_9',
               'green_draw_two', 'green_reverse', 'green_skip',
               'red_0', 'red_1', 'red_2', 'red_3', 'red_4', 'red_5', 'red_6', 'red_7', 'red_8', 'red_9', 'red_draw_two', 'red_reverse', 'red_skip',
               'wild', 'wild_draw_four',
               'yellow_0', 'yellow_1', 'yellow_2', 'yellow_3', 'yellow_4', 'yellow_5', 'yellow_6', 'yellow_7', 'yellow_8', 'yellow_9',
               'yellow_draw_two', 'yellow_reverse', 'yellow_skip']

# Function to make a prediction based on a file
def run_inference_file(image_path):
    # Convert image to RBG to ensure consistency for model input
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) # Data Augmentation
    # Using the model to make a prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Function to make a prediction based on video frames
def run_inference_video(frame):
    # Converting Video Frame to RBG
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0) # Data Augmentation
    # Using the model to make a prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class_name = class_names[predicted.item()]
    return predicted_class_name

class resize_window:
    def __init__(self, window, width, height):
        self.window = window
        self.width = width
        self.height = height

    def center_gui(self):
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x = (screen_w - self.width) // 2
        y = (screen_h - self.height) // 2
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")

def onclick_file():
    file_path = filedialog.askopenfilename(title="Select a file",
                                           filetypes=(("Image files", "*.jpg"), ("All files", "*.*")))

    if file_path:
        print(f"File selected: {file_path}")
        predicted_class_index = run_inference_file(file_path)
        predicted_class_name = class_names[predicted_class_index]

        image = Image.open(file_path)
        images = np.array(image)
        plt.figure(figsize=(6, 6))
        plt.imshow(images)
        plt.axis('off')
        plt.text(0.5, 1.05, f'Predicted Card: {predicted_class_name} ', ha='center', va='bottom', fontsize=15, color='black', weight='bold',
                 transform=plt.gca().transAxes)
        plt.show()

    root.withdraw()
    sub = tk.Tk()
    sub_window = resize_window(sub, 1000, 500)
    sub_window.center_gui()

    def close_file():
        root.deiconify()
        sub.destroy()

    sub.protocol("WM_DELETE_WINDOW", close_file)

def onclick_camera():
    root.withdraw()
    sub = tk.Toplevel()

    sub_window = resize_window(sub, 1000, 500)
    sub_window.center_gui()

    global label
    label = ttk.Label(sub)
    label.pack(padx=10, pady=10)

    back_button = tk.Button(sub, text="Back", command=lambda: close_camera(sub), activebackground="lightgray", activeforeground="red", anchor="center", bd=3, bg="lightblue",
                            cursor="hand2", disabledforeground="gray", font=("Arial", 12), height=2,
                            highlightbackground="black", highlightcolor="green", highlightthickness=3,
                            justify="center", overrelief="raised", padx=15, pady=5, width=8, wraplength=100)
    back_button.place(x=850, y=280)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    def show_frame():
        ret, frame = cap.read()
        if ret:
            predicted_class_name = run_inference_video(frame)
            cv2.putText(frame, f'Predicted Card: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)
            label.photo = photo
            label.configure(image=photo)

        sub.after(20, show_frame)

    show_frame()

    def close_camera(sub_window):
        root.deiconify()
        cap.release()
        cv2.destroyAllWindows()
        sub_window.destroy()

root = tk.Tk()
main_gui = resize_window(root, 1000, 500)
main_gui.center_gui()

modify_font = font.Font(family="MS Serif", size=18, weight="bold")
modify_font2 = font.Font(family="MS Serif", size=15)

frame = ttk.Frame(root)

label = ttk.Label(root, text="UNO Card Dectection System", font=modify_font, foreground="darkblue")
label.pack(padx=5, pady=8)

label2 = ttk.Label(root, text="Choose an option", font=modify_font2, foreground="green", borderwidth=2, width=15,
                   relief="groove", justify="center")
label2.pack(padx=5, pady=95)

sizegrip = ttk.Sizegrip(frame)
sizegrip.pack(expand=True, fill=tk.BOTH, anchor=tk.SE)
frame.pack(padx=3, pady=2, expand=True, fill=tk.BOTH)

file_button = tk.Button(root, text="Choose Image File", command=onclick_file, activebackground="lightgray", activeforeground="red", anchor="center", bd=3, bg="lightblue", cursor="hand2",
                        disabledforeground="gray", font=("Arial", 12), height=2, highlightbackground="black", highlightcolor="green", highlightthickness=3,
                        justify="center", overrelief="raised", padx=15, pady=5, width=15, wraplength=100)
file_button.place(x=200, y=225)

camera_button = tk.Button(root, text="Start Camera", command=onclick_camera, activebackground="lightgray", activeforeground="red", anchor="center", bd=3, bg="lightblue", cursor="hand2",
                          disabledforeground="gray", font=("Arial", 12), height=2, highlightbackground="black", highlightcolor="green", highlightthickness=3,
                          justify="center", overrelief="raised", padx=15, pady=5, width=15, wraplength=100)
camera_button.place(x=625, y=225)

root.mainloop()
