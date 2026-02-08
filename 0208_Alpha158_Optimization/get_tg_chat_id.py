
import requests
import time

TOKEN = "8052185621:AAFT1gMhEvxZYTixeijsjLA29Q6fpnEc1xs"

def get_chat_id():
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    print(f"Checking for messages to bot {TOKEN[:10]}...")
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if not data['ok']:
            print(f"Error: {data}")
            return
            
        results = data['result']
        if not results:
            print("❌ No messages found. Please send 'Hello' to your bot on Telegram (@ethtrend10mxiaodingdang_bot) and run this script again.")
            return
            
        # Get latest message
        latest = results[-1]
        chat_id = latest['message']['chat']['id']
        username = latest['message']['chat'].get('username', 'Unknown')
        
        print("\n✅ Found Chat ID!")
        print(f"Username: @{username}")
        print(f"Chat ID: {chat_id}")
        print("\nPlease copy this Chat ID and paste it into live_inference.py")
        
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    get_chat_id()
