import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Spiralfunktion
def create_spiral(num_points=500, noise_level=0, color='blue'):
    theta = np.linspace(0, 4 * np.pi, num_points)
    
    r = 0.1 * theta  
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x += np.random.normal(scale=noise_level, size=num_points)
    y += np.random.normal(scale=noise_level, size=num_points)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color=color, s=10)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    plt.title('Interaktiver Spiral Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

# Input Noise und Farbe
noise_slider = widgets.FloatSlider(value=0, min=0, max=0.15, step=0.01, description='Rauschen')
color_picker = widgets.ColorPicker(value='blue', description='Farbe')

def update_plot(noise_level, color):
    create_spiral(noise_level=noise_level, color=color)

widgets.interact(update_plot, noise_level=noise_slider, color=color_picker)
