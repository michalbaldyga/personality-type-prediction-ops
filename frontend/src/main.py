import customtkinter as ctk
from main_app_window import MainAppWindow

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = MainAppWindow()
    app.mainloop()
