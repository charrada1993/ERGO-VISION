# app.py
# Optimized for NVIDIA Jetson Orin reComputer J3011 (8 GB RAM, USB 2.0).
# Single OAK-D camera only. All tuning constants live in JetsonConfig.
import os
import threading
import depthai as dai
from config import Config, JetsonConfig
Config.ensure_dirs()


def _mem_used_mb() -> float:
    """Return current RSS memory usage in MB (reads /proc/meminfo)."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.strip().split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) / 1024
    except Exception:
        return 0.0

from camera.manager import CameraManager
from camera.mock_manager import MockCameraManager
from camera.imu_manager import IMUManager
from camera.calibration import CameraCalibration
from pose.estimator import PoseEstimator
from pose.fusion import PoseFusion
from pose.skeleton import SkeletonBuilder
from ergonomics.rula import RULACalculator
from ergonomics.reba import REBACalculator
from data.logger import DataLogger
from web.routes import create_app
from web.socket_events import SocketEvents


def main():
    # ── 1. Detect devices ─────────────────────────────────────────────────
    devices_info = dai.Device.getAllAvailableDevices()
    force_sim = os.environ.get('FORCE_SIMULATION', '0') == '1'

    if not devices_info or force_sim:
        print("[Main] No OAK-D device found or Simulation forced. Entering SIMULATION MODE.")
        use_mock = True
    else:
        use_mock = False

    # Single camera only — USB 2.0 cannot sustain more than one OAK-D
    devices_info = devices_info[:JetsonConfig.MAX_CAMERAS]   # MAX_CAMERAS = 1
    num_cams = len(devices_info)
    print(f"[Main] Using {num_cams} OAK-D device (USB 2.0 mode, single-camera)")

    cam_managers = []
    devices      = []
    calibrations = []

    if use_mock:
        cam_mgr = MockCameraManager()
        cam_mgr.setup()
        cam_mgr.start_streams()
        cam_managers.append(cam_mgr)
        calibrations.append(None)
        num_cams = 1
    else:
        # ── 2. Create pipelines and initialize cameras ───────────────────────
        for idx, dev_info in enumerate(devices_info):
            dev_id = getattr(dev_info, 'getMxId', getattr(dev_info, 'getDeviceId', lambda: getattr(dev_info, 'name', 'Unknown')))()
            print(f"[Main] Initializing device {idx+1}/{num_cams}: {dev_id}")
            pipeline = dai.Pipeline()
            cam_mgr = CameraManager(pipeline=pipeline)
            
            cam_mgr.setup()   # RGB + StereoDepth (aligned)
            # Note: IMU logic removed per vision-only mode requirement
            
            try:
                # Explicitly set HIGH speed for USB 2.0 stability on Jetson
                device = dai.Device(pipeline, dev_info, dai.UsbSpeed.HIGH)
            except Exception as e:
                print(f"[Main] Pipeline start failed for device {dev_info.getMxId()}: {e}")
                continue

            calib = CameraCalibration.from_device(device)
            cam_mgr.device = device
            cam_mgr.start_streams()
            
            devices.append(device)
            cam_managers.append(cam_mgr)
            calibrations.append(calib)

    if not cam_managers:
        print("[Main] Failed to initialize any devices. Exiting.")
        return

    # Update num_cams based on successfully initialized devices
    num_cams = len(cam_managers)

    # ── 2b. Visual IMU (optical-flow based, disabled to solve dashboard lag) ──────
    # imu_mgr = IMUManager()
    # imu_mgr.setup()
    # imu_mgr.set_camera_manager(cam_managers[0])   # uses primary camera RGB frames
    # imu_mgr.start()
    imu_mgr = None
    print("[Main] Visual IMU disabled (CPU optimization)")

    # ── 3. Pose components ────────────────────────────────────────────────
    pose_est   = PoseEstimator()
    pose_fusion = PoseFusion(num_cams)
    skeleton   = SkeletonBuilder()

    # ── 4. Ergonomic calculators ──────────────────────────────────────────
    rula_calc = RULACalculator()
    reba_calc = REBACalculator()

    # ── 5. Data logger ────────────────────────────────────────────────────
    logger = DataLogger()
    # Session is started dynamically via Socket.IO events

    # ── 6. Flask app and SocketIO ────────────────────────────────────────
    app, socketio = create_app()

    # ── 7. Store shared objects in app config ────────────────────────────
    # In multi-camera mode, we store the list of managers.
    app.config['CAMERA_MANAGERS'] = cam_managers
    app.config['CAMERA_MANAGER']  = cam_managers[0] # Primary camera for video feed compatibility
    app.config['CAMERA_MODE']     = num_cams
    app.config['CALIBRATION']     = calibrations[0] if calibrations else None
    app.config['IMU_MANAGER']     = imu_mgr

    # ── 8. Socket event handler ──────────────────────────────────────────
    socket_events = SocketEvents(
        socketio, cam_managers, pose_est, pose_fusion,
        skeleton, rula_calc, reba_calc, logger, app
    )

    # ── 9. Background processing thread ─────────────────────────────────
    processing_thread = threading.Thread(
        target=socket_events.process_loop, daemon=True
    )
    processing_thread.start()

    # ── 10. Run web server ────────────────────────────────────────────────
    # use_reloader=False: avoids double-process on Jetson (wastes ~200 MB RAM)
    # log_output=False:   silences per-request logs (saves ~5% CPU at 8 fps)
    print("[Main] ─────────────────────────────────────────────")
    print(f"[Main] RAM used: {_mem_used_mb():.0f} MB / 7607 MB")
    print("[Main] Dashboard → http://0.0.0.0:5000")
    print("[Main] ─────────────────────────────────────────────")
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False,
                     use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)
    finally:
        print("[Main] Shutting down …")
        if imu_mgr:
            imu_mgr.stop()
        for cam_mgr in cam_managers:
            cam_mgr.stop()
        for device in devices:
            device.close()
        print("[Main] Done.")


if __name__ == '__main__':
    main()