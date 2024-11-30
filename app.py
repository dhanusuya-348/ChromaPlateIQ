import base64
import numpy as np  
from flask import Flask, render_template, request, jsonify
import cv2
import os
from number_plate import process_plate, detect_plate_color

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.is_json:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
    else:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        image_path = os.path.join('Plates', file.filename)
        file.save(image_path)
        img = cv2.imread(image_path)

    plate_text = process_plate(img)
    plate_type = detect_plate_color(img)

    return jsonify({'plate_text': plate_text, 'plate_type': plate_type})

if __name__ == '__main__':
    app.run(debug=True)
