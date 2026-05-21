import pyaudio
p = pyaudio.PyAudio()

print("可用裝置列表：")

for i in range(p.get_device_count()):

    device_info = p.get_device_info_by_index(i)

    if device_info.get('maxInputChannels') > 0:

        print(f"Index {i}: {device_info.get('name')}")