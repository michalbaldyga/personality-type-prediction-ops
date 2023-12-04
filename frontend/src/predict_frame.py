import random
import webbrowser
from tkinter import filedialog

import customtkinter as ctk
from backend.predict.text.predict import predict as predict_text
from backend.predict.img.predict import predict as predict_image


class PredictFrame(ctk.CTkFrame):
    """Frame for displaying prediction results.

    Attributes:
        back_callback (function): Callback function to go back to the previous frame.
        method (str): The method used for prediction.
        segmented_button_var (tk.StringVar): Variable for the segmented button selection.
        segmented_button (ctk.CTkSegmentedButton): Segmented button for selecting different options.
        selected_label (ctk.CTkLabel): Label to display the selected option.
        save_button (ctk.CTkButton): Button for saving the prediction data to a file.
        hyperlink_label (ctk.CTkLabel): Hyperlink label to open a web browser.
    """

    def __init__(self, master, back_callback, method, data):
        """Initialize the PredictFrame.

        :param master: tk.Tk or tk.Toplevel, The master widget.
        :param back_callback: function, The callback function to go back to the previous frame.
        :param method: str, The method used for prediction.
        :param data: data (text/image) received from user.
        """
        super().__init__(master, width=400, height=300)
        self.back_callback = back_callback
        self.method = method
        self.data = data

        coins, category_name = self.get_names()

        # Segmented button
        self.segmented_button_var = ctk.StringVar(value=coins[0])
        self.segmented_button = ctk.CTkSegmentedButton(
            self,
            values=coins,
            command=self.segmented_button_callback,
            variable=self.segmented_button_var,
            height=30,
            width=300,
            dynamic_resizing=False,
        )
        self.segmented_button.pack(side="top", pady=(10))

        # Label to display the selected option
        self.selected_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.selected_label.pack(pady=8)

        # Create a frame for the buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(side="bottom", pady=10, padx=10, anchor="center")

        # Try again button
        ctk.CTkButton(button_frame, text="Try again", command=self.back, width=100, height=40, font=("Arial", 14)).pack(
            side="left", padx=10)

        # Save button
        self.save_button = ctk.CTkButton(button_frame, text="Save", command=self.save, width=100,
                                         height=40, font=("Arial", 14))
        self.save_button.pack(side="left", padx=10)

        # Set default value to "Human" and update the label
        self.segmented_button_var.set(coins[0])
        self.segmented_button_callback(coins[0])

        # Hyperlink at the bottom, centered
        self.hyperlink_label = ctk.CTkLabel(self, text="OPS Type Analyzer", font=("Arial", 14),
                                            cursor="hand2",
                                            text_color="#1F6AA5")
        self.hyperlink_label.bind("<Button-1>", lambda _: self.open_browser(
            "http://app.subjectivepersonality.com/analyzer?m=FF&s1=Fe&s2=Se&a=PCSB"))
        self.hyperlink_label.pack(side="bottom")

    def back(self):
        """Go back to the previous frame."""
        if self.back_callback:
            # Go back to the previous frame
            self.back_callback()
            self.destroy()

    def save(self):
        """Save the prediction data to a file."""
        coins, category_name = self.get_names()
        categories, values = self.get_data()

        # Create a string with the formatted data
        result_str = f"Predicted coins (method for prediction -> {self.method}):\n"

        beginning_of_human_coins = 0
        beginning_of_letter_coins = 3
        beginning_of_animal_coins = 7
        beginning_of_sexual_coins = 9

        for i, category in enumerate(categories):
            if i == beginning_of_human_coins:
                result_str += f"\n{coins[0]} coins:\n"
            if i == beginning_of_letter_coins:
                result_str += f"\n{coins[1]} coins:\n"
            if i == beginning_of_animal_coins:
                result_str += f"\n{coins[2]} coins:\n"
            if i == beginning_of_sexual_coins:
                result_str += f"\n{coins[3]} coins:\n"

            result_str += f"'{category_name[i]}': {category} {values[i]}%\n"

        # Ask the user for the file name and location
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")],
                                                 initialfile=[f"prediction_{self.method}.txt"])

        # Check if the user clicked "Cancel" or closed the dialog
        if not file_path:
            print("Save operation canceled.")
            return

        # Save the data to the selected file
        with open(file_path, "w") as file:
            file.write(result_str)

        print(f"Data saved to {file_path}")

    def open_browser(self, url):
        """Open a web browser with the specified URL."""
        webbrowser.open(url)

    @staticmethod
    def find_result(data, label):
        for element in data:
            if element['label'] == label:
                return element['percent']
        raise Exception("Invalid label")

    def read_coin_results(self, data, coin_group):
        result_coin1 = self.find_result(data, coin_group[0])
        result_coin2 = self.find_result(data, coin_group[1])

        if result_coin1 > result_coin2:
            return coin_group[0], result_coin1
        elif result_coin2 > result_coin1:
            return coin_group[1], result_coin2
        else:
            return f"{coin_group[0]}/{coin_group[1]}", result_coin1

    def get_data(self):
        """Generate random data for the prediction."""
        coin_groups = [["Oi", "Oe"], ["Di", "De"], ["DD", "OO"],
                       ["S", "N"], ["F", "T"], ["Sleep", "Play"],
                       ["Consume", "Blast"], ["Info", "Energy"],
                       ["Intro", "Extro"], ["Fem_S", "Mas_S"],
                       ["Fem_De", "Mas_De"]]

        results = {}

        if self.method == "selfie":
            results = predict_image(self.data)
        elif self.method == "link":
            results = predict_text(self.data)

        categories = []
        values = []

        for group in coin_groups:
            cat, val = self.read_coin_results(results, group)

            categories.append(cat.replace('_De', '').replace('_S', ''))
            values.append(val)

        return categories, values

    def get_names(self):
        """Get names for different coins and categories."""
        coins = ["Human", "Letter", "Animal", "Sexual"]

        category_name = [
            "Observer", "Decider", "Preferences", "Observer", "Decider",
            "Energy Animal", "Info Animal", "Dominant Animal",
            "Introverted vs Extraverted", "Sensory", "Extraverted Decider",
        ]

        return coins, category_name

    def segmented_button_callback(self, value):
        """Callback function for the segmented button."""
        print("Segmented button clicked:", value)

        coins, category_name = self.get_names()
        categories, values = self.get_data()

        # Update the label text based on the selected option
        if value == f"{coins[0]}":
            text = (
                f"'{category_name[0]}': {categories[0]} {values[0]}%\n\n"
                f"'{category_name[1]}': {categories[1]} {values[1]}%\n\n"
                f"'{category_name[2]}': {categories[2]} {values[2]}%"
            )
            cat = [categories[0], categories[1], categories[2]]
            val = [values[0], values[1], values[2]]
        elif value == f"{coins[1]}":
            text = (
                f"'{category_name[3]}': {categories[3]} {values[3]}%\n\n"
                f"'{category_name[4]}': {categories[4]} {values[4]}%"
            )
            cat = [categories[3], categories[4]]
            val = [values[3], values[4]]
        elif value == f"{coins[2]}":
            text = (
                f"'{category_name[5]}': {categories[5]} {values[5]}%\n\n"
                f"'{category_name[6]}': {categories[6]} {values[6]}%\n\n"
                f"'{category_name[7]}': {categories[7]} {values[7]}%\n\n"
                f"'{category_name[8]}': {categories[8]} {values[8]}%"
            )
            cat = [categories[5], categories[6], categories[7], categories[8]]
            val = [values[5], values[6], values[7], values[8]]
        elif value == f"{coins[3]}":
            text = (
                f"'{category_name[9]}': {categories[9]} {values[9]}%\n\n"
                f"'{category_name[10]}': {categories[10]} {values[10]}%"
            )
            cat = [categories[9], categories[10]]
            val = [values[9], values[10]]

        # Set the label text
        self.selected_label.configure(text=text)

        return cat, val
