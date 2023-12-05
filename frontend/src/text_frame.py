import customtkinter as ctk
from predict_frame import PredictFrame
from backend.predict.text.predict import predict

class StyledText(ctk.CTkFrame):
    """A styled text widget with customizable appearance.

    Attributes:
        textbox (ctk.CTkTextbox): The text widget.
    """

    def __init__(self, master=None, **kwargs):
        """Initialize the StyledText widget.

        :param master: tk.Tk or tk.Toplevel, The master widget.
        :param kwargs: Additional keyword arguments for customization.
        """
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
        """Insert text at the specified index.

        :param index: str, The index where the text should be inserted.
        :param text: str, The text to be inserted.
        :param tags: str or tuple, optional, Tags to apply to the inserted text.
        """
        self.textbox.insert(index, text, tags)

    def delete(self, index1, index2=None):
        """Delete text between the specified indices.

        :param index1: str, The starting index for deletion.
        :param index2: str, optional, The ending index for deletion. If not provided, deletes only the character at index1.
        """
        self.textbox.delete(index1, index2)

    def get(self, index1, index2=None):
        """Get the text between the specified indices.

        :param index1: str, The starting index.
        :param index2: str, optional, The ending index. If not provided, returns the character at index1.

        :return: str, The text between the specified indices.
        """
        return self.textbox.get(index1, index2)

    def configure(self, **kwargs):
        """Configure the appearance and behavior of the text widget.

        :param kwargs: Additional keyword arguments for configuration.
        """
        self.textbox.configure(**kwargs)

    def bind(self, sequence=None, func=None, add=None):
        """Bind a function to an event sequence.

        :param sequence: str, optional, The event sequence.
        :param func: function, optional, The function to bind to the event.
        :param add: str, optional, Specifies whether the new binding should be added or replace any existing binding.

        Note:
            Refer to Tkinter documentation for event sequence format.
        """
        self.textbox.bind(sequence, func, add)


class TextFrame(ctk.CTkFrame):
    """Frame for entering custom text for prediction.

    Attributes:
        back_callback (function): Callback function to go back to the previous frame.
        textbox (StyledText): The styled text widget.
    """

    def __init__(self, master=None, back_callback=None):
        """Initialize the TextFrame.

        :param master: tk.Tk or tk.Toplevel, The master widget.
        :param back_callback: function, The callback function to go back to the previous frame.
        """
        super().__init__(master, width=400, height=300)
        self.back_callback = back_callback

        self.textbox = StyledText(self)
        self.textbox.pack(pady=(20, 10))

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
        """Perform prediction with the custom text and show the PredictFrame."""
        show_text_frame = self.back_callback
        method = "text"

        custom_text = self.textbox.get("1.0", "end-1c")

        if custom_text:
            print(f"Predicting with custom text: {custom_text}")

            # Hide the current frame
            self.pack_forget()
            prediction_result = predict(custom_text)
            # Open the PredictFrame
            self.predict_frame = PredictFrame(self.master, show_text_frame, method, prediction_result)
            self.predict_frame.pack_propagate(False)
            self.predict_frame.pack(pady=(50, 10))
        else:
            print("No custom text provided")
