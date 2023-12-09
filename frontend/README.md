# Personality Type Prediction - frontend

This project is a Python-based GUI application that utilizes machine learning to predict a user's personality type based on their digital activity. The application provides a graphical user interface for users to interact with and input their digital data for analysis. Using advanced machine learning algorithms, the application processes the input data and generates predictions about the user's personality type.

## Packages

The following packages are used in this project:

- **`customtkinter`**: A custom GUI library built on top of `tkinter`, enhancing its functionality and aesthetic appeal. To check if `customtkinter` is installed, run the following command in the terminal:
  ```shell
  python3 -m customtkinter
  ```
    If it's not installed, you can install it using the following command:
  ```shell
  pip3 install customtkinter
  ```
   
- `tkinter`: A standard Python package for creating GUI applications. It is typically included with Python installations. To check if `tkinter` is installed on your system, run the following command in the terminal or command prompt:
    ```shell
    python3 -m tkinter
    ```
    If it's not installed, you can install it using the following command:
    ```shell
    pip3 install tkinter
    ```
- `Pillow`: An imaging library that extends Python Imaging Library (PIL) capabilities. It is used for image manipulation within the GUI.
    ```shell
    pip3 install Pillow
    ```
- `webbrowser`: A standard Python module facilitating the opening of hyperlinks or documentation from within the application.

## Project structure

Here is a list of the files and a short description of each:

- `main.py`:
  
  The main entry point of the application.
  Orchestrates the overall execution of the program.
  Handles the initialization of the main application window.
  
- `main_app_window.py`:

  Defines the main application window and its functionalities.
  Acts as the central hub for user interactions and navigation between different frames.
  
- `selfie_frame.py`:
  
  Implements the frame for choosing a selfie for prediction.
  Provides functionality for users to select an image and displays it on the frame.
  Includes a button to initiate the prediction process.

- `text_frame.py`:
  
  Manages the frame for entering custom text for prediction.
  Features a text entry widget where users can input text data.
  Includes buttons for navigation and initiating the prediction process.

- `link_frame.py`:
  
  Handles the frame for entering links for prediction.
  Allows users to input hyperlinks for analysis.
  Contains buttons for navigation and initiating the prediction process.

- `predict_frame.py`:
  
  Defines the prediction frame and its functionalities.
  Processes the selected input (selfie, text, link) for personality prediction.
  Displays the prediction results to the user.


## Application interface

![frontend](https://github.com/michalbaldyga/personality-type-prediction-ops/assets/105732925/6dc471c2-3b85-4c51-904a-7535e953248a)
