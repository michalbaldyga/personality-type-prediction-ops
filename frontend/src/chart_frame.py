import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import customtkinter as ctk


class ChartFrame(ctk.CTkFrame):
    def __init__(self, master, categories, values, selected_option):
        super().__init__(master, width=400, height=300)

        # Back button to close the chart frame
        back_button = ctk.CTkButton(self, text="Back", command=self.back, width=100, height=40, font=('Arial', 16))
        back_button.pack(side="bottom", anchor="se", pady=(0, 10), padx=10)

        # Create a sample bar chart
        self.create_chart(categories, values, selected_option)

    def back(self):
        self.destroy()

    def create_chart(self, categories, values, selected_option):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size

        # Remove the upper and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set y-axis limit to 100
        ax.set_ylim(0, 100)

        # Create a bar chart with smaller width
        bars = ax.bar(categories, values, width=0.5)

        # Set labels and title with larger fontsize
        ax.set_xlabel(f"{selected_option} " + "coins", fontsize=16)
        ax.set_ylabel('Probability', fontsize=16)
        ax.set_title('Predicted coins', fontsize=18)

        # Adjust padding from the bottom and around the plot
        plt.subplots_adjust(bottom=0.15, top=0.9)

        # Set the tick parameters to make ticks and labels larger
        ax.tick_params(axis='both', which='both', labelsize=14)

        # Display the chart in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1, pady=20, padx=20)

        # Display percentage values above each bar with larger fontsize
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}%",  # Format the percentage value
                ha="center",
                va="bottom" if height > 0 else "top",  # Adjust the position based on the bar's height
                fontsize=14,  # Adjust the font size
            )
