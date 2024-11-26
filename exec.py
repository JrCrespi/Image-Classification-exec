import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import torch
import torchvision.transforms as transforms

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Imagens")

        # Define the window size
        self.root.geometry("1280x720")

        try:
            # Load classification model
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pt')
            self.model = YOLO(model_path)
        except ImportError as e:
            print(f"Error importing YOLO or other dependencies: {e}")
            return

        # Transformation for the input image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.open_image_button = tk.Button(self.main_frame, text="Abrir Imagem", command=self.open_image)
        self.open_image_button.pack(pady=10)

        self.open_folder_button = tk.Button(self.main_frame, text="Abrir Pasta de Imagens", command=self.open_folder)
        self.open_folder_button.pack(pady=10)

        # Image frame
        self.image_frame = tk.Frame(root)

        self.back_button = tk.Button(self.image_frame, text="Voltar ao Menu Principal", command=self.show_main_menu)
        self.back_button.pack(pady=10)

        # Canvas for image with scrollbars
        self.canvas = tk.Canvas(self.image_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_y = tk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.scrollbar_x = tk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.config(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Label to display classification result
        self.result_label = tk.Label(self.image_frame, text="", font=("Arial", 12))
        self.result_label.pack(side=tk.RIGHT, anchor=tk.NE, padx=10, pady=10)

        self.image_list = []
        self.current_image_index = 0

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])

        if file_path:
            self.display_image(file_path)

    def open_folder(self):
        folder_path = filedialog.askdirectory()

        if folder_path:
            self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if self.image_list:
                self.current_image_index = 0
                self.display_image(self.image_list[self.current_image_index])
                self.show_image_navigation_buttons()

    def display_image(self, image_path):
        self.main_frame.pack_forget()
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Load image
        original_image = Image.open(image_path)

        # Get the dimensions of the window
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        # Calculate the aspect ratio of the image
        image_width, image_height = original_image.size
        aspect_ratio = image_width / image_height

        # Calculate the new size of the image to fit in the window while maintaining aspect ratio
        if image_width > window_width or image_height > window_height:
            if window_width / image_width < window_height / image_height:
                new_width = window_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = window_height
                new_width = int(new_height * aspect_ratio)
        else:
            new_width = image_width
            new_height = image_height

        # Resize the image
        resized_image = original_image.resize((new_width, new_height))

        # Convert the resized image to PhotoImage
        self.image = ImageTk.PhotoImage(resized_image)

        # Display the resized image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Perform classification
        self.run_classification(image_path)

    def run_classification(self, image_path):
        self.result = self.model.predict(image_path)
        
        for r in self.result:
            names = r.names
            top5 = r.probs.top5
            top5conf = r.probs.top5conf
            self.top5_names = [names[i] for i in top5]
            self.top5_confidences = top5conf
            self.update_classification_label()

    def update_classification_label(self):
        text = ""
        for name, conf in zip(self.top5_names, self.top5_confidences):
            if conf > 0.8:
                text += f"{name}: {conf.item()*100:.2f}%\n"
            else:
                text += f"{name}: {conf.item()*100:.2f}%\n"
        self.result_label.config(text=text)

    def on_canvas_resize(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def show_main_menu(self):
        self.image_frame.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def show_image_navigation_buttons(self):
        # Remove os botões de navegação existentes
        for widget in self.image_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.destroy()

        navigation_frame = tk.Frame(self.image_frame)
        navigation_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        prev_button = tk.Button(navigation_frame, text="Imagem Anterior", command=self.show_previous_image)
        prev_button.pack(side=tk.TOP, pady=5)

        next_button = tk.Button(navigation_frame, text="Próxima Imagem", command=self.show_next_image)
        next_button.pack(side=tk.TOP, pady=5)

    def show_next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_image(self.image_list[self.current_image_index])

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.image_list[self.current_image_index])

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
