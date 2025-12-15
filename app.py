from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera

app = Flask(__name__)

# Single global camera instance
camera = VideoCamera()

@app.route('/')
def index():
    headings = ("Name", "Album", "Artist")
    data = camera.df1.to_dict(orient='records') if hasattr(camera, "df1") and camera.df1 is not None else []
    return render_template('index.html', headings=headings, data=data)


def gen(camera):
    while True:
        try:
            frame_bytes, df = camera.get_frame()

            camera.df1 = df

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
            )

        except Exception as e:
            print("Error in video stream:", e)
            continue


@app.route('/video_feed')
def video_feed():
    return Response(
        gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/t')
def table_feed():
    if not hasattr(camera, "df1") or camera.df1 is None:
        return jsonify([])
    return jsonify(camera.df1.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
