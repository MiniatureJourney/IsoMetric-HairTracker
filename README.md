# IsoMetric-HairTracker
This project is a self-hosted computer vision application that aligns sequential images to a fixed baseline in order to enable consistent visual comparison over time. The system is designed to reduce visual variation caused by camera angle, rotation, and positioning by applying feature-based image alignment before storing or displaying images.

To use this project, first clone the repository to your local machine and ensure that Python 3.8 or higher is installed. Install the required dependencies using pip, including Flask for the web server, OpenCV for image processing, NumPy for matrix operations, and flask-cors to allow browser communication with the backend. Once dependencies are installed, start the application by running python app.py. This will launch a local web server on port 5000.

After starting the server, open a browser and navigate to http://localhost:5000. The frontend will request access to the device camera and display a live preview. The application supports three independent capture angles: front, left 45 degrees, and right 45 degrees. Each angle maintains its own baseline image and scan history.

For each angle, the first image captured is stored as the baseline. This baseline acts as the reference frame for all future comparisons. When a new image is captured, it is sent to the backend as a base64-encoded JPEG along with the selected angle. The backend decodes the image and converts it to grayscale for processing.

The alignment process begins by detecting keypoints in both the new image and the baseline image using the AKAZE feature detector. Binary descriptors are computed for each keypoint and matched using brute-force Hamming distance. The best matches are filtered based on descriptor distance, and a minimum number of valid matches is required to proceed. If sufficient matches are available, a homography matrix is estimated using the RANSAC algorithm. This matrix represents the geometric transformation required to align the new image with the baseline.

If a valid homography is successfully computed, the new image is warped into the baseline’s coordinate space using perspective transformation. The aligned image is then saved locally and returned to the frontend for comparison. If alignment fails at any stage, the scan is rejected and the user is prompted to adjust positioning or lighting.

On the frontend, once an aligned image is received, the application displays an interactive comparison view. The baseline image and the aligned image are layered, and a touch-controlled slider allows the user to visually inspect differences between the two images. A semi-transparent ghost overlay of the baseline is also used during capture to help maintain consistent framing before taking a photo.

All images are stored locally on the filesystem under angle-specific directories. No data is uploaded to external services, and no user accounts or cloud storage are required. The system is designed to be fully self-contained and privacy-preserving.

This implementation can be adapted for other use cases that require consistent visual comparison over time, such as fitness tracking, posture analysis, or rehabilitation progress monitoring.
