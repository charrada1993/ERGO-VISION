# web/routes.py
# Optimized for Jetson Orin reComputer J3011 over USB 2.0.
# MJPEG streams use JetsonConfig dimensions and quality settings.
from flask import Flask, render_template, jsonify, Response, send_file
from flask_socketio import SocketIO
from config import Config, JetsonConfig
import os
import cv2
import time
import numpy as np


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(Config.BASE_DIR, 'web', 'templates'),
        static_folder=os.path.join(Config.BASE_DIR, 'web', 'static')
    )
    app.config['SECRET_KEY'] = 'ergosecret!'
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    # Page routes
    @app.route('/')
    def dashboard():
        return render_template('dashboard.html')

    @app.route('/camera')
    def camera_page():
        return render_template('camera.html')

    @app.route('/rula')
    def rula_page():
        return render_template('rula.html')

    @app.route('/reba')
    def reba_page():
        return render_template('reba.html')

    @app.route('/3d')
    def threed_page():
        return render_template('3d.html')

    @app.route('/collection')
    def collection_page():
        return render_template('collection.html')

    @app.route('/report')
    def report_page():
        return render_template('report.html')

    # API routes
    @app.route('/api/config')
    def api_config():
        mode = app.config.get('CAMERA_MODE', 0)
        return jsonify({
            'mode':   mode,
            'usb3':   False,   # USB 2.0 connection
            'imu':    False,
            'has_rv': False,
        })

    @app.route('/api/sessions')
    def api_sessions():
        os.makedirs(Config.SESSION_DIR, exist_ok=True)
        files = [f for f in os.listdir(Config.SESSION_DIR) if f.endswith('.csv')]
        return jsonify({'sessions': files})

    @app.route('/api/generate_report/<filename>')
    def api_generate_report(filename):
        csv_path = os.path.join(Config.SESSION_DIR, filename)
        if not os.path.exists(csv_path):
            return "File not found", 404

        from reporting.report_generator import ReportGenerator
        try:
            pdf_path = ReportGenerator.generate(csv_path)
            return send_file(pdf_path, as_attachment=True,
                             download_name=os.path.basename(pdf_path))
        except Exception as e:
            return f"Error generating report: {str(e)}", 500

    # RGB MJPEG stream — USB 2.0 optimized
    # Rate: JetsonConfig.CAMERA_FPS (8 fps)
    # Size: JetsonConfig.VIDEO_WIDTH x VIDEO_HEIGHT (416x320)
    # Quality: JetsonConfig.VIDEO_JPEG_QUALITY (55)
    @app.route('/video_feed')
    def video_feed():
        cam_mgr = app.config.get('CAMERA_MANAGER')
        if not cam_mgr:
            return "Camera not available", 404

        def generate():
            _interval = 1.0 / JetsonConfig.CAMERA_FPS   # ~0.125 s at 8 fps
            _w = JetsonConfig.VIDEO_WIDTH
            _h = JetsonConfig.VIDEO_HEIGHT
            _q = JetsonConfig.VIDEO_JPEG_QUALITY
            while True:
                frames = cam_mgr.get_latest_frames()
                frame  = frames.get('rgb') if frames else None
                if frame is not None:
                    # Resize only if camera output differs from stream size
                    if frame.shape[1] != _w or frame.shape[0] != _h:
                        frame = cv2.resize(frame, (_w, _h),
                                           interpolation=cv2.INTER_LINEAR)
                    ret, jpeg = cv2.imencode('.jpg', frame,
                                             [cv2.IMWRITE_JPEG_QUALITY, _q])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + jpeg.tobytes() + b'--frame\r\n')
                time.sleep(_interval)

        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # Depth colourmap MJPEG stream — 5 Hz, small size to save CPU
    # Rate: JetsonConfig.DEPTH_STREAM_FPS (5 Hz)
    # Size: JetsonConfig.DEPTH_WIDTH x DEPTH_HEIGHT (320x180)
    # Quality: JetsonConfig.DEPTH_JPEG_QUALITY (50)
    @app.route('/depth_feed')
    def depth_feed():
        cam_mgr = app.config.get('CAMERA_MANAGER')
        if not cam_mgr:
            return "Camera not available", 404

        def generate():
            _interval = 1.0 / JetsonConfig.DEPTH_STREAM_FPS   # 0.2 s at 5 Hz
            _w = JetsonConfig.DEPTH_WIDTH
            _h = JetsonConfig.DEPTH_HEIGHT
            _q = JetsonConfig.DEPTH_JPEG_QUALITY
            while True:
                frames = cam_mgr.get_latest_frames()
                # Prefer raw disparity; fallback to raw 16-bit depth
                frame = frames.get('disp') if frames else None
                if frame is not None:
                    if len(frame.shape) == 2:
                        # Colourise raw disparity on-demand
                        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                else:
                    raw = frames.get('depth') if frames else None
                    if raw is not None:
                        norm  = cv2.normalize(raw, None, 0, 255,
                                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        frame = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)

                if frame is not None:
                    frame = cv2.resize(frame, (_w, _h),
                                       interpolation=cv2.INTER_NEAREST)
                    ret, jpeg = cv2.imencode('.jpg', frame,
                                             [cv2.IMWRITE_JPEG_QUALITY, _q])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + jpeg.tobytes() + b'--frame\r\n')
                time.sleep(_interval)

        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app, socketio