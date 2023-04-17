# Application for Video Retrieval

This code is a Python script that implements an image search engine using two different techniques: CLIP (Contrastive Language-Image Pre-Training) and 3D Histogram matching. The program uses the tkinter library to build a graphical user interface for the application.

The application has a main window with a text entry box and a search button. The user can enter a search query in the text box, and the application will display the top matching images in the form of buttons. The user can click on any of the images to select it, and then click on one of the three action buttons to perform an action on the selected image:

- "Histogram Similar" - displays images that are similar to the selected image using a 3D histogram matching technique.
- "CLIP Similar" - displays images that are similar to the selected image using the CLIP model.
- "Show Video" - displays images before and after the selected image.

The application displays up to 64 images at a time and provides a "Show More" button to display the next 96 images starting from the selected image.

**Usage:**
1. Store images in the "data/Images"
2. Precompute the descriptor vectors for your images using CLIP and store them in the "data/CLIP_VITB32.csv" file. 
3. Run the code.
