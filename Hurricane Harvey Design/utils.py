import os, io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ipywidgets as widgets

from ipywidgets import interact
from sklearn.metrics import confusion_matrix
from PIL import Image as ImageOps, ImageEnhance
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import plotly.graph_objects as go

import pickle
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import random

from typing import List, Dict, Callable


def data_aug_flip(image: tf.Tensor):
    '''Interactive function to illustrate flipping of images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''    
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")

    def _data_aug_flip(random_flip='horizontal'):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")

        if random_flip == 'horizontal':
            img2 = img.transpose(0)
        if random_flip == 'vertical':
            img2 = img.transpose(1)
        if random_flip == 'horizontal_and_vertical':
            img2 = img.transpose(0)
            img2 = img2.transpose(1)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Flipped')
        plt.axis("off")

    random_flip_widget = widgets.RadioButtons(
        options=['horizontal', 'vertical', 'horizontal_and_vertical'],
        value='horizontal',
        layout={'width': 'max-content'}, # If the items' names are long
        description='Random Flip',
        disabled=False,
        style = {'description_width': 'initial'},
    )
    
    interact(_data_aug_flip, random_flip = random_flip_widget)
    
    
def data_aug_zoom(image: tf.Tensor):
    '''Interactive function to illustrate zooming images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")

    def _data_aug_zoom(zoom_factor):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        w, h = img.size
        zoom2 = zoom_factor * 2
        x, y = int(w / 2), int(h / 2)
        img2 = img.crop(( x - w / zoom2, y - h / zoom2, 
                        x + w / zoom2, y + h / zoom2))

        img2 = img2.resize((w, h), ImageOps.LANCZOS)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Zoomed')
        plt.axis("off")

    zoom_widget = widgets.FloatSlider(
        value=1,
        min=1,
        max=2,
        step=0.05,
        description='Zoom: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'}
    )
    interact(_data_aug_zoom, zoom_factor = zoom_widget)


def data_aug_rot(image: tf.Tensor):
    '''Interactive function to illustrate rotating images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")
    
    def _data_aug_rot(angle):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        
        img2 = img.rotate(angle)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Rotated')
        plt.axis("off")

    angle_widget = widgets.FloatSlider(
        value=0,
        min=-40,
        max=40,
        step=5,
        description='Rotation (deg): ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'},

    )
    interact(_data_aug_rot, angle = angle_widget)


def data_aug_brightness(image: tf.Tensor):
    '''Interactive function to illustrate contrasting images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")
    enhancer = ImageEnhance.Brightness(img)

    def _data_aug_brightness(brightness_factor):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        
        img2 = enhancer.enhance(brightness_factor)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Brightness')
        plt.axis("off")

    brightness_widget = widgets.FloatSlider(
        value=1,
        min=0.5,
        max=1.5,
        step=0.2,
        description='Brightness factor: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'},
    )
    interact(_data_aug_brightness, brightness_factor = brightness_widget)





def get_performance_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> List:
    '''Calculates the Accuracy, Precision and Recall for the given pair of labels and prediction scores
    
    Args:
        y_true (np.ndarray): an array of true y values
        y_scores (np.ndarray): an array of predicted y values
        
    Returns:
        _ (List[float]): [accuracy, precision, recall]
    '''
    y_pred_1 = (y_scores >= 0.5) * 1
    return [accuracy_score(y_true, y_pred_1),
            precision_score(y_true, y_pred_1),
            recall_score(y_true, y_pred_1)]



    
    
d

def display_confusion_matrix(y_labels: np.ndarray, y_predict_prob_1: np.ndarray):
    '''Display a confusion matrix
    
    Input:
        y_labels (np.ndarray): An array like with the true lables
        y_predict_prob_1 (np.ndarray): An array like with the predictions
    '''
    confusion_matrix_1 = tf.math.confusion_matrix(
        y_labels.reshape(-1),
        (y_predict_prob_1.reshape(-1) >= 0.5)*1, # Convert probabilities to 0 and 1 labels
        num_classes=2
    ).numpy()

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_1,
                                  display_labels=[ 'No damage', 'Visible damage'])

    disp.plot(cmap="Blues", values_format='')
    
    

