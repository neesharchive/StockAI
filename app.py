import tkinter as tk
from tkinter import ttk
# Import the function to create the market graph
# If you're including the market graph directly in main.py, this import is not needed.
# from market_graph import create_market_graph

# Assuming the create_market_graph function is defined within this file for simplicity
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

def create_market_graph(parent):
    fig = Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    canvas = FigureCanvasTkAgg(fig, master=parent)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class StockMarketApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Stock Market Dashboard')
        self.root.geometry('800x600')  # Set your desired size

        # Apply a theme
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 'clam' is a simple and modern theme, experiment with others like 'alt', 'default', or 'classic'
        
        self.create_widgets()

    def create_widgets(self):
        # Profile Button
        self.profile_button = ttk.Button(self.root, text='View Profile', command=self.view_profile)
        self.profile_button.pack(side=tk.TOP, pady=10)
        
        # Real-time Market Graph
        create_market_graph(self.root)
        
        # Button to Run Another Application
        self.run_app_button = ttk.Button(self.root, text='Run Another App', command=self.run_another_app)
        self.run_app_button.pack(side=tk.BOTTOM, pady=10)
        
        # Additional UI elements can be added here
        
    def view_profile(self):
        # Placeholder for profile viewing functionality
        print("Profile viewing functionality goes here.")
    
    def run_another_app(self):
        # Placeholder for running another application
        print("Code to run another application goes here.")

if __name__ == '__main__':
    root = tk.Tk()
    app = StockMarketApp(root)
    root.mainloop()
