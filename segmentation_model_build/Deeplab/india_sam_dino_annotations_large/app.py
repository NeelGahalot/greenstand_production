import os
import random
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

SAMPLES_DIR = '/home/neel/mnt_data/greenstand/old_deeplab/segmentation_model_build/Deeplab/india_sam_dino_annotations_large/samples'
MASKS_DIR = '/home/neel/mnt_data/greenstand/old_deeplab/segmentation_model_build/Deeplab/india_sam_dino_annotations_large/masks'

APPROVED_FILE = '/home/neel/mnt_data/greenstand/old_deeplab/segmentation_model_build/Deeplab/india_sam_dino_annotations_large/approved.json'

@app.route('/imagelist')
def get_image_list():
    images = [f for f in os.listdir(SAMPLES_DIR) if f.endswith('.jpg')]
    return jsonify(images)

@app.route('/image/<filename>')
def get_image(filename):
    path = os.path.join(SAMPLES_DIR, filename)
    return send_file(path, mimetype='image/jpeg')

@app.route('/mask/<filename>')
def get_mask(filename):
    base = os.path.splitext(filename)[0]
    mask_path = os.path.join(MASKS_DIR, f"{base}_binarymask.jpg")
    return send_file(mask_path, mimetype='image/jpeg')

@app.route('/approve', methods=['POST'])
def approve():
    data = request.get_json()
    filename = data['filename']

    approved = []
    if os.path.exists(APPROVED_FILE):
        with open(APPROVED_FILE, 'r') as f:
            approved = json.load(f)

    if filename not in approved:
        approved.append(filename)
        with open(APPROVED_FILE, 'w') as f:
            json.dump(approved, f, indent=2)

    return jsonify({"message": f"{filename} approved!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
