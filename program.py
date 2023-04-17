import tkinter as tk
from tkinter import ttk
 
import requests
from PIL import Image, ImageTk
import os
import numpy as np
import torch
import clip
import csv

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def cosineDistance (v1, v2):
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if(d == 0):
        return 2
    return 1 - np.dot(v1, v2) / d

def hist_3D(img, quant):
    img_pixels = np.asarray(img.resize((30,30)))

    xyz = np.zeros((quant+1, quant+1, quant+1))

    for i in range(img_pixels.shape[0]):
        for j in range(img_pixels.shape[1]):
            (r, g, b) = img_pixels[i, j]//quant
            xyz[r, g, b] += 1

    hist = xyz.flatten()

    return hist
 
shown = 64
url = "https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php"
dataset_path = "data/Images/" 

filenames = []
for fn in sorted(os.listdir(dataset_path)):
    filename = dataset_path + fn
    filenames.append(filename)

# Read CLIP vectors 
with open("data/CLIP_VITB32.csv", "r") as f:
    lines = f.readlines()

vector_list = []
for line in lines:
    numbers = line.strip().split(";")
    vector = [float(x) for x in numbers]
    vector_list.append(vector)


# Downloadig a CLIP model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
 
# Create histogram vectors 
histograms = []
for file in filenames:
    histograms.append(hist_3D(Image.open(file) , 16))


# Create an application

root = tk.Tk()
root.title("Searcher")
root.wm_attributes('-fullscreen', 'true')
 
image_size = (int(root.winfo_screenwidth() / 9.5) - 12, int(root.winfo_screenheight() / 8) - 4)

images_buttons = []
selected_images = []
shown_images = []
current_order = []


def hide_borders():
    global selected_images
    for button in images_buttons:
        button.config(bg="red")
    selected_images = []
 
 
def search_clip(text):
    print("Using CLIP for query: ", text)
    text = clip.tokenize(text).to(device)
    text_feature = model.encode_text(text)
    text_feature = text_feature.reshape(text_feature.shape[1])

    probs_per_image = []
    with torch.no_grad():
        for vector in vector_list:
            probs_per_image.append(cosineDistance(text_feature, vector))

    top_result = np.argsort(probs_per_image)
    current_order = top_result

    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[top_result[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[top_result[i]],
                                    command=(lambda j=i: on_click(j)))
    hide_borders()

def histogram_update():
    print(f"Showing Hist similar images for image: {selected_images[0]}")
    choosen_image = Image.open(selected_images[0])
    choosen_hist = hist_3D(choosen_image, 16)

    probs_per_image = []
    for hist in histograms:
        probs_per_image.append(cosineDistance(choosen_hist, hist))
    
    top_result = np.argsort(probs_per_image)
    print(np.sort(probs_per_image))

    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[top_result[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[top_result[i]],
                                    command=(lambda j=i: on_click(j)))
    hide_borders()

def clip_update():
    print(f"Showing CLIP similar images for image: {selected_images[0]}")
    choosen_feature = model.encode_image(preprocess(Image.open(selected_images[0])).unsqueeze(0).to(device))
    
    probs_per_image = []
    with torch.no_grad():
        for vector in vector_list:
            probs_per_image.append(cosineDistance(choosen_feature, vector)[0])

    top_result = np.argsort(probs_per_image)

    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[top_result[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[top_result[i]],
                                    command=(lambda j=i: on_click(j)))
    hide_borders()
 
def show_video():
    print(f"Showing video for image: {selected_images[0]}")
    index = (selected_images[0][-9:])[:5].lstrip('0')

    count = int(index) - 41

    selected = [count + i for i in range(shown)]

    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[selected[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[selected[i]],
                                    command=(lambda j=i: on_click(j)))

def show_more():
    print(f"Show more 96 pics starting from image: {selected_images[0]}")
    index = (selected_images[0][-9:])[:5].lstrip('0')

    selected = [int(index) + i for i in range(shown)]

    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[selected[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[selected[i]], command=(lambda j=i: on_click(j)))

def show_less():
    print(f"Show less 96 pics starting from image: {selected_images[0]}")
    index = (selected_images[0][-9:])[:5].lstrip('0')
    count = int(index) - 95

    selected = [count + i for i in range(shown)]

    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[selected[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[selected[i]], command=(lambda j=i: on_click(j)))


def on_click(index):
    if images_buttons[index].cget("bg") == "yellow":
        images_buttons[index].config(bg="black")
        selected_images.remove(images_buttons[index].cget("text"))
    else:
        images_buttons[index].config(bg="yellow")
        selected_images.append(images_buttons[index].cget("text"))
    text_index.config(text="Last selected image: " + selected_images[0][12:17])
 
 
def on_double_click():
    hide_borders()
 
def close_win(e):
    root.destroy()
 
def send_result():
    key_i = (selected_images[0][-9:])[:5]
    my_obj = {'team': "eklipina", 'item': key_i}
 
    x = requests.get(url=url, params=my_obj, verify=False)
    print(x.text)



# Create window
window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
window.pack(fill=tk.BOTH, expand=True)
 
# Create frames
search_bar = ttk.Frame(window, width=root.winfo_screenwidth() / 4, height=root.winfo_screenheight(), relief=tk.SUNKEN)
result_frame = ttk.Frame(window, width=(3 * root.winfo_screenwidth()) / 4, height=root.winfo_screenheight(),
                         relief=tk.SUNKEN)
window.add(search_bar, weight=1)
window.add(result_frame, weight=4)
 
# Add text input
tk.Label(search_bar, text="Text query:").pack(side=tk.TOP, pady=5)
text_input = tk.Entry(search_bar, bd=3, width=32)
text_input.bind("<Enter>", (lambda: search_clip(text_input.get())))
text_input.pack(side=tk.TOP, pady=5)
 

# Search button
clip_button = tk.Button(search_bar, text="Search Clip", command=(lambda: search_clip(text_input.get())))
clip_button.pack(side=tk.TOP)
 
# Selected image
text_index = tk.Label(search_bar, text="Selected image: ")
text_index.pack(side=tk.TOP, pady=5)

# Show more button
show_more_button = tk.Button(search_bar, text="Show more", command=(lambda: show_more()))
show_more_button.pack(side=tk.TOP)

# Show less button
show_less_button = tk.Button(search_bar, text="Show less", command=(lambda: show_less()))
show_less_button.pack(side=tk.TOP)


# Histogram similarity update button
histogram_similarity_button = tk.Button(search_bar, text="Histogram similarity", command=(lambda: histogram_update()))
histogram_similarity_button.pack(side=tk.TOP)

# CLIP similarity update button
CLIP_similarity_button = tk.Button(search_bar, text="CLIP similarity", command=(lambda: clip_update()))
CLIP_similarity_button.pack(side=tk.TOP)

# show video button
show_video_button = tk.Button(search_bar, text="Show video", command=(lambda: show_video()))
show_video_button.pack(side=tk.TOP)
 
# Sending select result
send_result_b = tk.Button(search_bar, text="Send selected index", command=(lambda: send_result()))
send_result_b.pack(side=tk.TOP, pady=5)
# set control-v to set result
root.bind('<Control-v>', lambda e: send_result())
 
# Set images
for s in range(shown):
    # load image
    shown_images.append(ImageTk.PhotoImage(Image.open(filenames[s]).resize(image_size)))
    # create button
    images_buttons.append(tk.Button(result_frame, bg="black", bd=2, text=filenames[s], image=shown_images[s],
                                    command=(lambda j=s: on_click(j))))
    # set position of button
    images_buttons[s].grid(row=(s // 8), column=(s % 8), sticky=tk.W)
    # set double click to reset marking of images
    images_buttons[s].bind('<Double-1>', lambda event: on_double_click())
 
# Set escape as exit
root.bind('<Escape>', lambda e: close_win(e))
 
root.mainloop()