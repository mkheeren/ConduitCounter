from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

def count_pipes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not load image.", None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        count = len(circles)

        for (x, y, r) in circles:
            cv2.circle(image, (x, y), int(r * 0.8), (0, 255, 0), 3)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

        result_path = os.path.join(RESULT_FOLDER, "detected_pipes.png")
        cv2.imwrite(result_path, image)

        return count
    else:
        return 0

@app.route("/api/count-circles", methods=["POST"])
def api_count_circles():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image = request.files["image"]

    # Save uploaded image
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    image.save(upload_path)

    # Count circles
    circle_count = count_pipes(upload_path)

    return jsonify({"circle_count": circle_count})

@app.route("/result/<filename>")
def result_file(filename):
    result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    if not os.path.exists(result_path):
        return "Result not found", 404
    return send_file(result_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
