import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response


app = Flask(__name__)


model = YOLO('tiny_512_trained_pruned_0_5.pt')


# # stream=False
# def run_inference(frame):
#     results = model(frame)
#     annotated_frame = results[0].plot()
#     return annotated_frame


def run_inference(frame):
    results = model(frame, stream=True)
    for result in results:
        annotated_frame = result.plot()
        return annotated_frame


def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка открытия вебкамеры")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = run_inference(frame)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

