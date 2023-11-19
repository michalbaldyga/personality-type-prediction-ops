import customtkinter as ctk
from predict_frame import PredictFrame


class StyledText(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.textbox = ctk.CTkTextbox(self, wrap="word", font=("Arial", 14),
                                      fg_color=("#F9F9FA", "#343638"),
                                      border_color=("#979DA2", "#565B5E"),
                                      text_color=("gray10", "#DCE4EE"),
                                      corner_radius=6,
                                      border_width=2,
                                      insertwidth=4,
                                      height=200,
                                      width=300)
        self.textbox.pack(fill="both", expand=True)

    def insert(self, index, text, tags=None):
        self.textbox.insert(index, text, tags)

    def delete(self, index1, index2=None):
        self.textbox.delete(index1, index2)

    def get(self, index1, index2=None):
        return self.textbox.get(index1, index2)

    def configure(self, **kwargs):
        self.textbox.configure(**kwargs)

    def bind(self, sequence=None, func=None, add=None):
        self.textbox.bind(sequence, func, add)


class TextFrame(ctk.CTkFrame):
    def __init__(self, master=None, back_callback=None):
        super().__init__(master, width=400, height=300)
        self.back_callback = back_callback

        self.textbox = StyledText(self)
        self.textbox.pack(pady=(20, 10))

        # Create a frame for the buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(side="bottom", pady=10, padx=10, anchor="center")

        # Back button
        ctk.CTkButton(button_frame, text="Back", command=self.back, width=100, height=40, font=('Arial', 14)).pack(
            side="left", padx=10)

        # Predict button
        ctk.CTkButton(button_frame, text="Predict", command=self.predict, width=100, height=40,
                      font=('Arial', 14)).pack(side="right", padx=10)

    def back(self):
        if self.back_callback:
            self.back_callback()

    def predict(self):
        show_text_frame = self.back_callback
        method = "text"

        custom_text = self.textbox.get("1.0", "end-1c")

        if custom_text:
            print(f"Predicting with custom text: {custom_text}")

            # Hide the current frame
            self.pack_forget()

            # Open the PredictFrame
            self.predict_frame = PredictFrame(self.master, show_text_frame, method)
            self.predict_frame.pack_propagate(False)
            self.predict_frame.pack(pady=(50, 10))
        else:
            print("No custom text provided")
