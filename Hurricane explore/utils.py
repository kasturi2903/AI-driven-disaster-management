import ipywidgets as widgets

from ipywidgets import HBox
import folium 
from folium.plugins import FastMarkerCluster
import matplotlib.pyplot as plt
from PIL import Image
import os
import os
import numpy as np 
import pandas as pd

from IPython.display import clear_output, display
from ipyfilechooser import FileChooser

from typing import List, Dict, Tuple, Any, Callable







def interactive_plot_pair(base: str, matches: List[str]) -> Callable:
    '''Create a plot to visualize a pair of images at the same location. One
    showing damage and the other showing no damage.
    
    Args:
        base (str): The base of the image path
        matches (List[str]): a list of image names
    Returns:
        plot_image_pairs (Callable): A function that plots the image pairs given the image index
    '''
    def plot_pairs(base, matches, index):
        fig = plt.figure(figsize=(12, 12))
        ax = []

        im_damage = Image.open(os.path.join(base, 'visible_damage', matches[index])).resize((200, 200))
        im_nodamage = Image.open(os.path.join(base, 'no_damage', matches[index])).resize((200, 200))

        ax.append(fig.add_subplot(1, 2, 1))
        ax[-1].set_title("No damage") 
        ax[-1].axis('off')
        plt.imshow(im_nodamage)

        ax.append(fig.add_subplot(1, 2, 2))
        ax[-1].set_title("Visible damage") 
        ax[-1].axis('off')
        plt.imshow(im_damage)
        plt.axis('off')
        plt.show()


    def plot_image_pairs(file_index):
        plot_pairs(base, matches, index=file_index)
        
    return plot_image_pairs
    

def load_coordinates(path: str, samples: int) -> List[Tuple[str, Tuple[float, float]]]:
    '''Load the  GPS coordinates from the first few samples in a given folder
    
    Args:
        path (str): path to the images
        samples (int): number of samples to take
    
    Returns:
        coordinates: An array containing the GPS coordianates extracted from the filenames
    '''
    files = os.listdir(path)
    coordinates = []
    indexes = list(range(len(files)))
    np.random.shuffle(indexes)
    indexes = indexes[0:samples]
    for i in indexes:
        # Get the coordinates
        coordinate = files[i].replace('.jpeg', '').split('_')
        coordinates.append((files[i], (float(coordinate[1]) , float(coordinate[0]))))
        
    return coordinates


def get_dataframe_from_file_structure() -> pd.core.frame.DataFrame:
    ''' Creates a dataframe with metadata based on the file structure.
    
    Returns:
        _ (pd.core.frame.DataFrame): Dataframe with metadata
    '''
    # Dataset paths
    base = './data'
    subsets = ['train', 'validation', 'test']
    labels = ['visible_damage', 'no_damage']

    # Navigate through every folder and its contents to create a dataframe
    data = []
    for seti in subsets:
        for label in labels:
            files = os.listdir(os.path.join(base, seti, label))
            for filename in files:
                path = os.path.join(seti, label, filename)
                lon, lat = filename.replace(".jpeg", "").split("_")
                data.append([seti, label, lat, lon, path, filename])

    # Create dataframe
    return pd.DataFrame(data = data, columns=['subset', 'label', 'lat', 'lon', 'path', 'filename'])
