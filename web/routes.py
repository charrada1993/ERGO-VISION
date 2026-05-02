# web/routes.py
from flask import Flask, render_template, jsonify, Response, send_file
from flask_socketio import SocketIO
from config import Config
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

    # ─── Page routes ─────────────────────────────────────────────────
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

    # ─── API routes ───────────────────────────────────────────────────
    @app.route('/api/config')
    def api_config():
        mode     = app.config.get('CAMERA_MODE', 0)
        return jsonify({
            'mode':    mode,
            'usb3':    True,
            'imu':     False,
            'has_rv':  False,
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
            return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))
        except Exception as e:
            return f"Error generating report: {str(e)}", 500

    # ─── RGB MJPEG stream ─────────────────────────────────────────────
    @app.route('/video_feed')
    def video_feed():
        cam_mgr = app.config.get('CAMERA_MANAGER')
        if not cam_mgr:
            return "Camera not available", 404

        def generate():
            while True:
                frames = cam_mgr.get_latest_frames()
                frame  = frames.get('rgb') if frames else None
                if frame is not None:
                    ret, jpeg = cv2.imencode(
                        '.jpg', frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + jpeg.tobytes() + b'--frame\r\n')
                time.sleep(1.0 / 15)   # Match the 15 FPS camera rate

        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # ─── Depth colourmap MJPEG stream ────────────────────────────────
    @app.route('/depth_feed')
    def depth_feed():
        cam_mgr = app.config.get('CAMERA_MANAGER')
        if not cam_mgr:
            return "Camera not available", 404

        def generate():
            while True:
                frames = cam_mgr.get_latest_frames()
                # Prefer the pre-colourised disparity; fallback: raw depth
                frame = frames.get('disp') if frames else None
                if frame is not None:
                    # If it's a 1-channel raw disparity frame, colourise it now
                    if len(frame.shape) == 2:
                        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                else:
                    raw = frames.get('depth') if frames else None
                    if raw is not None:
                        # Fallback: normalise uint16 depth to uint8 for display
                        norm = cv2.normalize(raw, None, 0, 255,
                                             cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        frame = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)

                if frame is not None:
                    # Resize to a smaller fixed size for depth
                    frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_NEAREST)
                    ret, jpeg = cv2.imencode(
                        '.jpg', frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 60]
                    )
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + jpeg.tobytes() + b'--frame\r\n')
                time.sleep(1.0 / 10)   # Depth is heavy, update less frequently

        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app, socketio