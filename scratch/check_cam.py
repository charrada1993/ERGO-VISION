import depthai as dai
devices = dai.Device.getAllAvailableDevices()
print(f"Found {len(devices)} devices")
for dev in devices:
    print(f"Device: {dev.getMxId()}")
