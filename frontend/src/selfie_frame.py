from tkinter import filedialog

import customtkinter as ctk
from PIL import Image
from predict_frame import PredictFrame


class SelfieFrame(ctk.CTkFrame):
    """Frame for choosing a selfie for prediction.

    Attributes:
        back_callback (function): Callback function to go back to the previous frame.
        selected_image (str): Path to the selected image.
    """

    def __init__(self, master, back_callback):
        """Initialize the SelfieFrame.

        :param master: tk.Tk or tk.Toplevel, The master widget.
        :param back_callback: function, The callback function to go back to the previous frame.
        """
        super().__init__(master, width=400, height=300)
        self.back_callback = back_callback
        self.selected_image = None

        ctk.CTkLabel(self, text="Choose selfie for prediction", font=("Arial", 18)).pack(pady=(20, 2))

        # Create a frame for the buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(side="bottom", pady=10, padx=10, anchor="center")

        # Back button
        ctk.CTkButton(button_frame, text="Back", command=self.back, width=100, height=40, font=("Arial", 14)).pack(
            side="left", padx=10)

        # Select image button
        self.image_button = ctk.CTkButton(button_frame, text="Select image", command=self.select_image, width=100,
                                          height=40, font=("Arial", 14))
        self.image_button.pack(side="left", padx=10)

        # Predict button
        self.predict_button = ctk.CTkButton(button_frame, text="Predict", command=self.predict, width=100, height=40,
                                            font=("Arial", 14))
        self.predict_button.pack(side="right", padx=10)

        # Default image
        default_image_path = "face_landmark.png"
        self.default_image = Image.open(default_image_path)
        ctk_image = ctk.CTkImage(light_image=self.default_image, dark_image=self.default_image, size=(170, 170))

        # Image label
        self.image_label = ctk.CTkLabel(self, text="", image=ctk_image, font=("Arial", 14))
        self.image_label.pack(side="top", pady=10, padx=10)

    def back(self):
        """Go back to the previous frame."""
        if self.back_callback:
            self.back_callback()

    def select_image(self):
        """Open a file dialog to select an image and display it."""
        image_path = filedialog.askopenfilename(title="", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if image_path:
            self.selected_image = image_path
            self.display_selected_image(image_path)

    def display_selected_image(self, image_path):
        """Display the selected image on the frame.

        :param image_path: str, The path to the selected image.
        """
        try:
            pil_image = Image.open(image_path)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(170, 170))
            self.image_label.configure(image=ctk_image)

            # Keep a reference to prevent garbage collection
            self.image_label.image = ctk_image

            # Clear any existing text
            self.image_label.text = ""
        except Exception as e:
            print(f"Error displaying image: {e}")

    def predict(self):
        """Perform prediction with the selected image and show the PredictFrame."""
        show_selfie_frame = self.back_callback
        method = "selfie"

        # Print the path to the selected image
        if self.selected_image:
            print(f"Predicting image: {self.selected_image}")

            # Hide the current frame
            self.pack_forget()

            # Open the PredictFrame
            self.predict_frame = PredictFrame(self.master, show_selfie_frame, method, None)
            self.predict_frame.pack_propagate(False)
            self.predict_frame.pack(pady=(50, 10))
        else:
            print("No selfie image provided")
