
import requests

TG_TOKEN = "8052185621:AAFT1gMhEvxZYTixeijsjLA29Q6fpnEc1xs"
TG_CHAT_ID = "6290088209"

def test_send():
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    message = "🔔 **AI 机器人连接测试成功！**\n\n如果您收到这条消息，说明实盘推送配置正确。🚀"
    
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    print(f"Connecting to Telegram API...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        result = response.json()
        if result.get("ok"):
            print("✅ Success! Check your Telegram.")
        else:
            print(f"❌ Failed: {result}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_send()
