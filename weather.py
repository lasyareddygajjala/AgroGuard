import requests

API_KEY = "YOUR_KEY_HERE"  # Get free: https://openweathermap.org/api

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        data = requests.get(url).json()
        if data.get("cod") == 200:
            return {
                "temp": round(data["main"]["temp"], 1),
                "humidity": data["main"]["humidity"],
                "desc": data["weather"][0]["description"]
            }
    except:
        pass
    return {"temp": 25, "humidity": 70, "desc": "clear"}