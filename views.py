from flask import Blueprint, render_template
from flask_login import login_required, current_user

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io


views = Blueprint('views', __name__)


@views.route('/contact')
def contact():
    return render_template('contact.html')

@views.route('/faqs')
def faqs():
    return render_template('faqs.html')

@views.route('/')
@login_required 
def home():
    return render_template('index.html', user=current_user)



def detect_pollution(image):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    lower_pollution = np.array([20, 50, 50])
    upper_pollution = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_pollution, upper_pollution)
    pollution_percent = (np.sum(mask) / 255) / (mask.shape[0] * mask.shape[1]) * 100
    pollution_percent = 100 - pollution_percent
    return pollution_percent





@views.route('/analyze', methods= ['POST'])
@login_required
def analyze_process():
    file = request.files['image']
    img = Image.open(file.stream)
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    pollution_level = detect_pollution(img_array)
    return jsonify({
        'pollution_level': float(pollution_level),
        'status': 'High' if pollution_level > 50 else 'Moderate' if pollution_level > 25 else 'Low'
    })

@views.route('/analyze')
@login_required
def analyze():
    return render_template('analyze.html')

@views.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)