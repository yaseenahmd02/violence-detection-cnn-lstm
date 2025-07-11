# telegram_alert.py
import requests

def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("📨 Alert sent successfully.")
    else:
        print(f"❌ Failed to send alert: {response.text}")

# Example usage:
# send_telegram_alert("YOUR_BOT_TOKEN", "YOUR_CHAT_ID", "⚠️ Violence Detected in video!")
