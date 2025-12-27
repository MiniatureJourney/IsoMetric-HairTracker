import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Base directory for storing all image scans
UPLOAD_FOLDER = 'scans'

# Default user namespace (can be replaced with auth/session logic later)
DEFAULT_USER_ID = 'default_user'


def align_images(image, template):
    """
    Aligns an input image to a baseline template using
    AKAZE feature matching and homography.
    """
    im1_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    detector = cv2.AKAZE_create()
    kp1, desc1 = detector.detectAndCompute(im1_gray, None)
    kp2, desc2 = detector.detectAndCompute(im2_gray, None)

    if desc1 is None or desc2 is None:
        return None

    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    )
    matches = sorted(
        matcher.match(desc1, desc2, None),
        key=lambda x: x.distance
    )

    good_matches = matches[:int(len(matches) * 0.15)]
    if len(good_matches) < 20:
        return None

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if matrix is None:
        return None

    return cv2.warpPerspective(
        image,
        matrix,
        (template.shape[1], template.shape[0])
    )


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        angle = data.get('angle', 'front')

        user_id = data.get('user_id', DEFAULT_USER_ID)

        np_arr = np.frombuffer(
            base64.b64decode(image_data),
            np.uint8
        )
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        user_dir = os.path.join(UPLOAD_FOLDER, user_id, angle)
        os.makedirs(user_dir, exist_ok=True)

        baseline_path = os.path.join(user_dir, 'baseline.jpg')

        if not os.path.exists(baseline_path):
            cv2.imwrite(baseline_path, img)
            return jsonify({
                "status": "baseline_set",
                "message": "Baseline image saved"
            })

        baseline_img = cv2.imread(baseline_path)
        aligned_img = align_images(img, baseline_img)

        if aligned_img is None:
            return jsonify({
                "status": "error",
                "message": "Alignment failed. Please adjust angle or lighting."
            })

        scan_count = len([
            f for f in os.listdir(user_dir)
            if f.startswith('scan_')
        ]) + 1

        scan_path = os.path.join(
            user_dir,
            f'scan_{scan_count}.jpg'
        )
        cv2.imwrite(scan_path, aligned_img)

        _, buffer = cv2.imencode('.jpg', aligned_img)

        return jsonify({
            "status": "success",
            "message": "Scan saved successfully",
            "aligned_image": (
                "data:image/jpeg;base64," +
                base64.b64encode(buffer).decode('utf-8')
            )
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Server error occurred"
        })


@app.route('/history', methods=['GET'])
def get_history():
    angle = request.args.get('angle', 'front')
    user_id = request.args.get('user_id', DEFAULT_USER_ID)

    path = os.path.join(UPLOAD_FOLDER, user_id, angle)
    if not os.path.exists(path):
        return jsonify([])

    images = []
    for filename in sorted(os.listdir(path), reverse=True):
        if filename.endswith('.jpg'):
            with open(os.path.join(path, filename), 'rb') as img_file:
                encoded = base64.b64encode(
                    img_file.read()
                ).decode('utf-8')
                images.append({
                    "name": filename,
                    "data": "data:image/jpeg;base64," + encoded
                })

    return jsonify(images)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
