#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFIA Analyzer

@author: theanikolaou
updated: 04/02/2025

"""

import PySimpleGUI as sg
import subprocess
import os
import time
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # Import matplotlib for figure saving
from scipy.signal import hilbert
import pandas as pd
import scipy
from datetime import datetime
import csv

#set up
csv_file = None
writer = None 
current_date = datetime.now().strftime("%Y-%m-%d_%H")  # Format as YYYY-MM-DD_HH

csv_filename = f"/home/ericksonlab/Desktop/app_data/data_{current_date}.csv"  # Generate filename
calib_params = [[0.5395, 2.588, 0.7322, -0.01926],
                [0.3043,2.353, 2.212, 0.01717],
                [3.913, 4/945, 0.3862, -0.06876]]
img_profiles = pd.DataFrame()

#backend function definitions
def start_csv():
    # Create and open the CSV file
    global writer, csv_file
    csv_file = open(csv_filename, mode="w", newline="")
    writer = csv.writer(csv_file)
    #Write headers to the CSV file
    writer.writerow(["Image Name", "Control Intensity", "Test Intensity", "TC Ratio", "Concentration"])
    csv_file.flush()
    

def write_csv(imageName, concentration, TC_array):
    global writer,csv_file
    writer.writerow([imageName, TC_array[0], TC_array[1], TC_array[2], concentration]) 
    csv_file.flush()

def capture_img(imageName):
    LED_on = 'raspi-gpio set 24 dh'
    subprocess.call(LED_on, shell=True)
    capture_image = 'libcamera-still -t 2000 --ev -2 -e png -o /home/ericksonlab/Desktop/images/' + imageName + '.png'  
    subprocess.call(capture_image, shell=True)
    print(f"Saving to: /home/ericksonlab/Desktop/images/{imageName}.png")
    LED_off = 'raspi-gpio set 24 dl'
    subprocess.call(LED_off, shell=True)
    
def myHampel(vec, k, stds=10, minp=0):
    y = vec
    y.index = range(len(vec))
    k = (2 * k) + 1
    movmeds = y.rolling(window=k, min_periods=minp, center=True, closed='right').median()
    movmeans = y.rolling(window=k, min_periods=1).mean()
    scale = -1 / (np.sqrt(2) * scipy.special.erfinv(3/2))
    xsigma = scale * y
    outliers = ~(np.abs(y - movmeds) <= stds * xsigma)
    y[outliers] = movmeds[outliers]
    return y

def rgb2grey(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def TC_intensities(img_array): 
    
    #Vertical Crop
    image_height, image_width = np.shape(img_array)
    imageFile_height = int(image_height / 2)
    imageFile_width = int(image_width / 2)
    imageFile_vertical_cropped = img_array[img_array - 75:imageFile_height + 75, :]
    
    #horizontal crop
    imageFile_vertical_cropped_profile = np.average(imageFile_vertical_cropped, axis=0)
    imageFile_smoothed_profile = myHampel(pd.Series(imageFile_vertical_cropped_profile),5,2,5).rolling(window=5,min_periods=0,center=True,closed='right').mean().to_numpy()
    imageFile_smoothed_profile_hilbert = np.imag(hilbert((imageFile_smoothed_profile)))
    minp_control = np.argmin(imageFile_smoothed_profile_hilbert)
    imageFile_horizontal_cropped = imageFile_vertical_cropped [:,minp_control-150:minp_control+500]
    
    # Profile cropped
    imageFile_profile = np.average(imageFile_horizontal_cropped, axis=0)

    #filters
    imageFile_smoothed_profile = myHampel(pd.Series(imageFile_profile),5,2,5).rolling(window=5,min_periods=0,center=True,closed='right').mean().to_numpy()
    
    imageFile_smoothed_profile_hilbert = np.imag(hilbert((imageFile_smoothed_profile)))
    
    max_control = np.argmax(imageFile_smoothed_profile_hilbert)
    min_control = np.argmin(imageFile_smoothed_profile_hilbert)
    
    line_width = max_control-min_control
    
    min_test = min_control+ 300
    max_test = min_test+line_width 
    
    background_index = int(max_control + (min_test-max_control)/2)
    
    # Background corrected profile
    background_intensity = imageFile_smoothed_profile[background_index-2:background_index+2]
    background_intensity = np.average(background_intensity)
    background_intensity = round(background_intensity,3)

    background_profile = np.full(len(imageFile_smoothed_profile),background_intensity)
    corrected_imageFile_smoothed_profile = np.subtract(imageFile_smoothed_profile,background_profile)
    corrected_imageFile_smoothed_profile[corrected_imageFile_smoothed_profile < 0] = 0
    
    processed_img = corrected_imageFile_smoothed_profile

    
    # Calculate Intensity values
    # Control Intensity
    control_line = corrected_imageFile_smoothed_profile[min_control:max_control]
    control_peak = np.argmax(control_line)
    control_intensity = control_line[control_peak-2:control_peak+2]
    control_intensity = np.average(control_intensity)
    control_intensity = round(control_intensity,3)

    
    # Test Intensity
    test_line = corrected_imageFile_smoothed_profile[min_test:max_test]
    test_peak = np.argmax(test_line)
    test_intensity = test_line[test_peak-2:test_peak+2]
    test_intensity = np.average(test_intensity)
    test_intensity  = round(test_intensity ,3)

    
    return(processed_img, [test_intensity, control_intensity])

def process_img(image_path):
    image = Image.open(image_path).convert("RGB")
    grey_image = rgb2grey(image)
    img_array = np.rot90(np.array(grey_image), 1)
    
    #PR profile extraction
    PR_array = img_array[100:200, 150:300]
    PR_TC = TC_intensities(PR_array)
    
    #ER profile extraction 
    ER_array = img_array[300:400, 150:300]
    ER_TC = TC_intensities(ER_array)
    
    #HER2 profile extraction
    HER_array = img_array[500:600, 150:300]
    HER_TC = TC_intensities(HER_array)
    
    #Calculate TC ratios for each
    avg_control = np.average(PR_TC[1], ER_TC[1], HER_TC[1])
   
    TC_array = [PR_TC[0],ER_TC[0],HER_TC[0]]/avg_control
    
    return(img_array, TC_array)

def estimate_conc(parameters, TC_ratio):
    return(parameters[2]*np.power(((calib_params[0]-TC_ratio)/(TC_ratio-calib_params[3])), 1/calib_params[1]))

def analyze_img(calib_params, TC_array):

    est_conc = [estimate_conc(calib_params[0], TC_array[0]), 
                estimate_conc(calib_params[1], TC_array[1]), 
                estimate_conc(calib_params[2], TC_array[2])]
    
    #Luminal A
    if est_conc[0] > 100  and est_conc[1] > 100 and est_conc[2] < 23: 
        result = f"""Estimated PR: {est_conc[0]} pM
        Estimated ER: {est_conc[1]} pM
        Estimated HER2: {est_conc[2]} pM
        Subtype: Luminal A
        """
    #Luminal B    
    elif est_conc[0] > 50 and est_conc[0] < 150 and est_conc[1] > 68 and est_conc[1] < 200 and est_conc[2] > 23 and est_conc[2] < 60: 
        result = f"""Estimated PR: {est_conc[0]} pM
        Estimated ER: {est_conc[1]} pM
        Estimated HER2: {est_conc[2]} pM
        Subtype: Luminal B
        """
    #HER2 Enriched
    elif est_conc[0] < 30  and est_conc[1] < 68 and est_conc[2] > 60: 
        result = f"""Estimated PR: {est_conc[0]} pM
        Estimated ER: {est_conc[1]} pM
        Estimated HER2: {est_conc[2]} pM
        Subtype: HER2-enriched
        """
    #Triple Negative
    elif est_conc[0] < 30  and est_conc[1] < 68 and est_conc[2] < 23: 
        result = f"""Estimated PR: {est_conc[0]} pM
        Estimated ER: {est_conc[1]} pM
        Estimated HER2: {est_conc[2]} pM
        Subtype: Potential Triple-Negative 
        """
    #Error? 
    elif est_conc[:] < 0 and est_conc[:] == 'nan' and est_conc[:] == 'inf': 
        result = "Error!"
    

    return(est_conc, result)

#frontend definitions

#Page 1:
def capture_page():
    return[
        [sg.Text("Enter Sample ID:", font=("Helvetica", 20), justification='center', background_color="lightgray", text_color="navy")],
        [sg.Input(key='-INPUT-', size=(20, 8))],  # Input field
        [sg.Button('1', key='-NUM-1', font='default 20 bold', size=(8, 1), pad=(10, 10)),
         sg.Button('2', key='-NUM-2', font='default 20 bold', size=(8, 1), pad=(10, 10)),
         sg.Button('3', key='-NUM-3', font='default 20 bold', size=(8, 1), pad=(10, 10))],
        [sg.Button('4', key='-NUM-4', font='default 20 bold', size=(8, 1), pad=(10, 10)),
         sg.Button('5', key='-NUM-5', font='default 20 bold', size=(8, 1), pad=(10, 10)),
         sg.Button('6', key='-NUM-6', font='default 20 bold', size=(8, 1), pad=(10, 10))],
        [sg.Button('7', key='-NUM-7', font='default 20 bold', size=(8, 1), pad=(10, 10)),
         sg.Button('8', key='-NUM-8', font='default 20 bold', size=(8, 1), pad=(10, 10)),
         sg.Button('9', key='-NUM-9', font='default 20 bold', size=(8, 1), pad=(10, 10))],
        [sg.Button('0', key='-NUM-0', font='default 20 bold', size=(8, 1), pad=(10, 10))],
        [sg.Button('Delete', key='-DELETE-', font='default 20 bold', size=(10, 1), pad=(10, 20))],
        [sg.Button('Analyze Sample', font='default 20 bold', size=(15, 1))],
        [sg.Button('Exit', font='default 20 bold', size=(15, 1), pad=(10, 10))],
        [sg.Image(filename="/Users/ericksonlab/Documents/cornell_seal_resized.png", background_color="lightgray")]
        ]



window = sg.Window('LFIA Analyzer', capture_page(), element_justification='center', background_color="lightgray",
                    grab_anywhere=False, size=(480, 750),finalize=True,no_titlebar=False)
window.Maximize()

#initialize variables
imageFileName = ""
TC_array = []
start_csv()

while True:
    LED_state = 'raspi-gpio set 24 op'
    subprocess.call(LED_state, shell=True)
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Exit':
        csv_file.close()
        break

    #Input Buttons
    elif event.startswith('-NUM-'): # Handle buttons that start with '-NUM-'
       digit = event.split('-')[-1]  # Get the last part of the key to identify the number
       values['-INPUT-'] += digit  # Append the digit to the input
       window['-INPUT-'].update(values['-INPUT-'])  # Update the input field
    elif event == '-DELETE-':  # Handle delete button
       current_input = values['-INPUT-']
       values['-INPUT-'] = current_input[:-1]  # Remove the last character
       window['-INPUT-'].update(values['-INPUT-'])  # Update the input field

    #Image Capture
    elif event == 'Analyze Sample':
        imageFileName = values['-INPUT-']
        capture_img(imageFileName)
        img_array, TC_array = process_img('/home/reprophone/Desktop/images/' + imageFileName + '.png') #insert filepath specific to your raspi device
        est_conc, result = analyze_img(calib_params, TC_array)
        write_csv(imageFileName, est_conc, TC_array)
        sg.popup(result, title="Assay Result")


# Close the window
window.close()


