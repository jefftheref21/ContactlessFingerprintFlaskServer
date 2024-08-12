from flask import Flask, request, send_file
from PIL import Image
import base64
from io import BytesIO
from zipfile import ZipFile
import torch

from database import Database

import numpy as np
from lib import enhance_contactless, segmentation
from lib.Fingerprint_Matching import infer
import os
import cv2

app = Flask(__name__)
db = Database()

EMBEDDINGS_FOLDER = 'embeddings/'
if not os.path.exists(EMBEDDINGS_FOLDER):
    os.makedirs(EMBEDDINGS_FOLDER)

UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['EMBED_FOLDER'] = EMBEDDINGS_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_enhanced(segments1, segments2, hand1, hand2):
    enh_out1 = []
    for segs in segments1:
        segs = cv2.rotate(segs, cv2.ROTATE_90_CLOCKWISE)
        enh = enhance_contactless.main(segs)[1]
        if hand1.lower() == "left":
            enh = cv2.rotate(enh, cv2.ROTATE_90_CLOCKWISE)
        if hand1.lower() == "right":
            enh = cv2.rotate(enh, cv2.ROTATE_90_COUNTERCLOCKWISE)
        enh_out1.append(enh)

    enh_out2 = []        
    for segs in segments2:
        segs = cv2.rotate(segs, cv2.ROTATE_90_COUNTERCLOCKWISE)
        enh = enhance_contactless.main(segs)[1]
        if hand1.lower() == "left":
            enh = cv2.rotate(enh, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if hand1.lower() == "right":
            enh = cv2.rotate(enh, cv2.ROTATE_90_CLOCKWISE)
        enh_out2.append(enh)

    return enh_out1, enh_out2

def process(img_arr):
    img = Image.fromarray(img_arr)
    img = img.resize((1008, 1344), Image.ANTIALIAS)

    img_io = BytesIO()
    img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    #img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_io

@app.route('/', methods=['GET'])
def ping():
    return "Hi. You have successfully been pinged!"

@app.route('/enrollment', methods=['POST'])
def check_username():
    username = request.json['username']
    # Non-repeated username
    if db.check_valid_username(username):
        return {'message': 'Username is valid'}, 200
    return {'message': 'Username already exists'}, 422

@app.route('/enrollment/scan', methods=['POST'])
def enroll_credentials():
    if 'file1' not in request.files or 'file2' not in request.files:
        return {'message': 'No selected file'}, 400

    file1 = request.files['file1']
    file2 = request.files['file2']
    hand1 = "left"
    hand2 = "right"

    if file1.filename == '' or file2.filename == '':
        return {'message': 'No selected file'}, 400

    img1 = Image.open(file1.stream)
    img1_arr = np.array(img1)
    img2 = Image.open(file2.stream)
    img2_arr = np.array(img2)

    username = request.form.get('username')
    password = request.form.get('password')
    path1 = username + '_LEFT_' + file1.filename[4:12]
    path2 = username + '_RIGHT_' + file2.filename[4:12]

    bounding_box1, segments1 = segmentation.main(img1_arr, hand1)
    emb_1 = infer.get_embeddings(segments1)
    bounding_box2, segments2 = segmentation.main(img2_arr, hand2)
    emb_2 = infer.get_embeddings(segments2)

    if db.add_user(username, password, path1, path2):
        img1.save(os.path.join(app.config['UPLOAD_FOLDER'], path1 + '.png'))
        img2.save(os.path.join(app.config['UPLOAD_FOLDER'], path2 + '.png'))
        torch.save(emb_1, os.path.join(app.config['EMBED_FOLDER'], path1 + '.pt'))
        torch.save(emb_2, os.path.join(app.config['EMBED_FOLDER'], path2 + '.pt'))
        return {'message': 'File and fields uploaded successfully'}, 200

    return {'message': 'Upload to server was unsuccessful'}, 500

@app.route('/verification', methods=['POST'])
def verify_credentials():
    username = request.json['username']
    password = request.json['password']
    user = db.get_user(username, password)
    if user:
        return {'message': 'Credentials verified',
                'leftFingerprintPath': user['leftFingerprintPath'], 
                'rightFingerprintPath': user['rightFingerprintPath']
                }, 200
    return {'message': 'Incorrect username or password'}, 400

@app.route('/verification/scan', methods=['POST'])
def verify_fingerprints():
    if 'file1' not in request.files or 'file2' not in request.files:
        return {'message': 'No selected file'}, 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        return {'message': 'No selected file'}, 400
    
    hand1 = "left"
    hand2 = "right"

    img1 = Image.open(file1.stream)
    img1_arr = np.array(img1)
    img2 = Image.open(file2.stream)
    img2_arr = np.array(img2)
    
    bounding_box1, segments1 = segmentation.main(img1_arr, hand1)
    bounding_box2, segments2 = segmentation.main(img2_arr, hand2)
    
    inp_emb1 = infer.get_embeddings(segments1)
    inp_emb2 = infer.get_embeddings(segments2)

    enrolled1 = request.form.get('enrolled1')
    enrolled2 = request.form.get('enrolled2')

    img3 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], enrolled1 + '.png'))
    img3_arr = np.array(img3)
    img4 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], enrolled2 + '.png'))
    img4_arr = np.array(img4)

    bounding_box3, segments3 = segmentation.main(img3_arr, hand1)
    bounding_box4, segments4 = segmentation.main(img4_arr, hand2)

    enr_emb1 = torch.load(os.path.join(app.config['EMBED_FOLDER'], enrolled1 + '.pt'))
    enr_emb2 = torch.load(os.path.join(app.config['EMBED_FOLDER'], enrolled2 + '.pt'))

    sim1, sim_list1 = infer.compare_embeddings(enr_emb1, inp_emb1)
    sim2, sim_list2 = infer.compare_embeddings(enr_emb2, inp_emb2)

    score1 = sim1.item()
    if (score1 > 0.50):
        pred1 = "Match Found"
    else:
        pred1 = "Not a Match"

    score2 = sim2.item()
    if (score2 > 0.50):
        pred2 = "Match Found"
    else:
        pred2 = "Not a Match"

    enh1, enh2 = get_enhanced(segments1, segments2, hand1, hand2)
    enh3, enh4 = get_enhanced(segments3, segments4, hand1, hand2)

    images = []
    images.append((f'left_inp_bbox.jpg', process(bounding_box1)))
    images.append((f'right_inp_bbox.jpg', process(bounding_box2)))
    images.append((f'left_enr_bbox.jpg', process(bounding_box3)))
    images.append((f'right_enr_bbox.jpg', process(bounding_box4)))

    for i, enh in enumerate(enh1):
        images.append((f'left_inp_enh{i}.jpg', process(enh)))
    for i, enh in enumerate(enh2):
        images.append((f'right_inp_enh{i}.jpg', process(enh)))
    for i, enh in enumerate(enh3):
        images.append((f'left_enr_enh{i}.jpg', process(enh)))
    for i, enh in enumerate(enh4):
        images.append((f'right_enr_enh{i}.jpg', process(enh)))

    # Create a BytesIO buffer to hold the ZIP file
    buffer = BytesIO()

    # Create a ZIP file in memory
    with ZipFile(buffer, 'w') as zipf:
        for filename, img_io in images:
            zipf.writestr(filename, img_io.read())
            img_io.seek(0)  # Reset buffer position for the next read
    
    buffer.seek(0)
    zip_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return {'message': 'Successful verification',
            'left_score': score1,
            'right_score': score2,
            'left_pred': pred1,
            'right_pred': pred2,
            'left_sim_list': sim_list1, 
            'right_sim_list': sim_list2,
            'zip_file': zip_base64,
            }, 200

@app.route('/identification', methods=['POST'])
def identify():
    if 'file' not in request.files:
        return {'message': 'No selected file'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'message': 'No selected file'}, 400
    
    type = request.form.get('type')
    
    img = Image.open(file.stream)
    img_arr = np.array(img)
    bounding_box, segments = segmentation.main(img_arr, type)
    emb_2 = infer.get_embeddings(segments)
    
    highest_score = 0.0
    identified_username = ""
    users = db.get_all_users()
    for user in users:
        path = user['leftFingerprintPath'] if type == 'left' else user['rightFingerprintPath']
        emb_1 = torch.load(EMBEDDINGS_FOLDER + path + '.pt')
        print(user['username'])
        print(emb_1)
        sim, sim_list = infer.compare_embeddings(emb_1, emb_2)
        if sim.item() > highest_score:
            highest_score = sim.item()
            identified_username = user['username']
    print(highest_score)
    if highest_score < 0.5:
        return {'message': 'No identification found'}, 204

    return {'message': 'Successful Identification',
            'username': identified_username,
            'score': highest_score
            }, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
