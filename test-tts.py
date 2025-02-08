#!/usr/bin/env python3
import requests

def test_tts_api():
    # 设置 FastAPI 服务的 URL
    url = "http://127.0.0.1:8010/tts"
    
    # 构造请求数据，文本内容和语言参数（根据需要自行修改）
    payload = {
        "text": "Hello, world! This is a test.",
        "language": "en"
    }
    
    try:
        # 发送 POST 请求，内容以 JSON 格式提交
        response = requests.post(url, json=payload)
        
        # 检查响应状态码
        if response.status_code == 200:
            print("API调用成功：")
            print(response.json())
        else:
            print(f"API调用失败，状态码：{response.status_code}")
            print("错误信息：", response.text)
    except Exception as e:
        print("请求过程中发生异常：", e)

if __name__ == "__main__":
    test_tts_api()
