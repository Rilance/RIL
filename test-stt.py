import requests

# STT API 地址
API_URL = "http://127.0.0.1:8000/transcribe"

# 需要测试的音频文件路径（请替换为你的测试音频文件）
AUDIO_FILE_PATH = "audio/test_audio.wav"

def test_stt_api(audio_path):
    """测试 STT 语音识别 API"""
    with open(audio_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        result = response.json()
        print("识别结果:", result["text"])
    else:
        print("API 请求失败:", response.status_code, response.text)

if __name__ == "__main__":
    test_stt_api(AUDIO_FILE_PATH)
