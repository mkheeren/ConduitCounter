from flask import Flask, request, jsonify, render_template, send_file
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

# implementation heavily inspired by:
# https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
def count_pipes(image_path, min_radius=0, max_radius=0):
    print(min_radius)
    print(max_radius)

    # load the image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not load image.", None

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # save the gray and blurred images for demo
    grayscale_path = os.path.join("results", "grayscale_image.png")
    cv2.imwrite(grayscale_path, gray)
    blurred_path = os.path.join("results", "blurred_image.png")
    cv2.imwrite(blurred_path, blurred)

    #use HoughCircles to detect circles, using given radius
    # todo, find good value for dp, param1, param2. could possibly be adjusted to better detect
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(min_radius * 1.5),
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        # round circle coords to integers, required
        circles = np.round(circles[0, :]).astype("int")
        count = len(circles)

        # draw circles and center dot on the image 
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), (int)(r*.8), (0, 255, 0), 3)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1) 

        # save the result image
        result_path = os.path.join("results", "detected_pipes.png")
        cv2.imwrite(result_path, image)

        return count, result_path
    else:
        return 0, None

# https://flask.palletsprojects.com/en/stable/quickstart/#http-methods
@app.route("/", methods=["GET", "POST"])
def upload_and_detect():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # get the radius from the form, add room for variance
        radius = request.form.get('radius', type=int)
        min_radius = int(radius * .9)
        max_radius = int(radius * 1.1)

        # save uploaded file and count its pipes
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(input_path)
        count, result_path = count_pipes(input_path, min_radius, max_radius)

        if result_path:
            return jsonify({
                "message": f"Number of pipes detected: {count}",
                "result_image": f"/result/{os.path.basename(result_path)}"
            })
        else:
            return jsonify({"message": "No pipes detected."})

    # return the html on get
    return render_template("upload.html")

# return the result image
@app.route("/result/<filename>")
def result_file(filename):
    result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    return send_file(result_path, mimetype="image/png")

# expose to the local network, rather than just localhost
if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")
