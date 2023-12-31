import customtkinter as ctk
from PIL import Image
from predict_frame import PredictFrame
from backend.predict.get_transcript_from_url import get_transcript
from backend.predict.text.predict import predict


class LinkFrame(ctk.CTkFrame):
    """A frame for handling YouTube link input and prediction."""

    def __init__(self, master, back_callback):
        """Initialize the LinkFrame.

        :param master: The Tkinter master widget.
        :param back_callback: The callback function to go back to the previous frame.
        :return: None
        """
        super().__init__(master, width=400, height=300)
        self.back_callback = back_callback

        # Create a label
        ctk.CTkLabel(self, text="Please enter the YouTube link below", font=("Arial", 18)).pack(pady=20)

        # Create an entry with placeholder text
        self.link_entry = ctk.CTkEntry(self, placeholder_text="https://www.youtube.com/example", width=300, height=40,
                                       font=("Arial", 14))
        self.link_entry.pack(pady=10)

        # Default image
        default_image_path = "link-icon.png"
        self.default_image = Image.open(default_image_path)
        ctk_image = ctk.CTkImage(light_image=self.default_image, dark_image=self.default_image, size=(80, 80))

        # Image label
        self.image_label = ctk.CTkLabel(self, text="", image=ctk_image, font=("Arial", 14))
        self.image_label.pack(side="top", pady=10, padx=10)

        # Create a frame for the buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(side="bottom", pady=10, padx=10, anchor="center")

        # Back button
        ctk.CTkButton(button_frame, text="Back", command=self.back, width=100, height=40, font=("Arial", 14)).pack(
            side="left", padx=10)

        # Predict button
        ctk.CTkButton(button_frame, text="Predict", command=self.predict, width=100, height=40,
                      font=("Arial", 14)).pack(side="right", padx=10)

    def back(self):
        """Go back to the previous frame."""
        if self.back_callback:
            self.back_callback()

    def predict(self):
        """Perform prediction based on the YouTube link."""
        show_link_frame = self.back_callback
        method = "link"

        youtube_link = self.link_entry.get()

        if youtube_link:
            print(f"Predicting with YouTube link: {youtube_link}")

            try:
                transcript = get_transcript(youtube_link)

                # Hide the current frame
                self.pack_forget()
                prediction_results = predict(transcript)

                # Open the PredictFrame
                self.predict_frame = PredictFrame(self.master, show_link_frame, method, prediction_results)
                self.predict_frame.pack_propagate(False)
                self.predict_frame.pack(pady=(50, 10))
            except Exception as e:
                print("An error occured in prediction: " + e)
        else:
            print("No YouTube link provided")
