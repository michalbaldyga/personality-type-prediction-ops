import customtkinter as ctk
from link_frame import LinkFrame
from selfie_frame import SelfieFrame
from text_frame import TextFrame
from predict_frame import PredictFrame
import webbrowser


class MainAppWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Personality Prediction App")
        self.geometry("600x400")

        # Create welcome frame
        self.welcome_frame = ctk.CTkFrame(
            self,
            width=400,
            height=300)
        self.welcome_frame.pack_propagate(False)

        ctk.CTkLabel(self.welcome_frame,
                     text="Welcome to Personality Prediction App",
                     font=('Arial', 18)
                     ).pack(pady=40)
        ctk.CTkLabel(self.welcome_frame,
                     text="You are invited to try our personality type prediction app based on digital activity. \n\n"
                          "We are using OPS model for prediction.\n",
                     font=('Arial', 14),
                     wraplength=300
                     ).pack(pady=20)

        ctk.CTkButton(self.welcome_frame, text="Let's start!", command=self.show_home_frame, width=100, height=40,
                      font=('Arial', 14)).pack(pady=10)

        # Create home frame
        self.home_frame = ctk.CTkFrame(self, width=400, height=300)
        self.home_frame.pack_propagate(False)
        ctk.CTkLabel(self.home_frame, text="Select the method for prediction", font=('Arial', 18)).pack(
            pady=20)

        ctk.CTkButton(self.home_frame, text="Selfie", command=self.show_selfie_frame, width=100, height=40,
                      font=('Arial', 14)).pack(pady=5)
        ctk.CTkButton(self.home_frame, text="YouTube link", command=self.show_link_frame, width=100, height=40,
                      font=('Arial', 14)).pack(pady=5)
        ctk.CTkButton(self.home_frame, text="Custom text", command=self.show_text_frame, width=100, height=40,
                      font=('Arial', 14)).pack(pady=5)
        ctk.CTkButton(self.home_frame, text="OPS self-test",
                      command=lambda: self.open_browser("https://v.lroy.us/ObjectivePersonalityTest/index.html"),
                      width=100,
                      height=40,
                      font=('Arial', 14),
                      cursor="hand2",
                      fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).pack(pady=15)

        # Create selfie frame
        self.selfie_frame = SelfieFrame(self, self.show_home_frame)
        self.selfie_frame.pack_propagate(False)

        # Create link frame
        self.link_frame = LinkFrame(self, self.show_home_frame)
        self.link_frame.pack_propagate(False)

        # Create text frame
        self.text_frame = TextFrame(self, self.show_home_frame)
        self.text_frame.pack_propagate(False)

        # Show the welcome frame initially
        self.show_welcome_frame()

        # Hyperlink at the bottom, centered
        self.hyperlink_label = ctk.CTkLabel(self, text="Learn more about OPS", font=('Arial', 14),
                                            cursor="hand2",
                                            text_color="#1F6AA5")
        self.hyperlink_label.bind("<Button-1>", lambda event: self.open_browser(
            "https://subjectivepersonality.wordpress.com/2020/08/19/what-is-ops/"))
        self.hyperlink_label.pack(side="bottom", pady=10)

    def show_welcome_frame(self):
        self.hide_frames()
        self.welcome_frame.pack(pady=(50, 10))

    def show_home_frame(self):
        self.hide_frames()
        self.home_frame.pack(pady=(50, 10))

    def show_selfie_frame(self):
        self.hide_frames()
        self.selfie_frame.pack(pady=(50, 10))

    def show_link_frame(self):
        self.hide_frames()
        self.link_frame.pack(pady=(50, 10))

    def show_text_frame(self):
        self.hide_frames()
        self.text_frame.pack(pady=(50, 10))

    def hide_frames(self):
        self.welcome_frame.pack_forget()
        self.home_frame.pack_forget()
        self.selfie_frame.pack_forget()
        self.link_frame.pack_forget()
        self.text_frame.pack_forget()

    def open_browser(self, url):
        webbrowser.open(url)
