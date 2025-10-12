import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Define constants directly
AQICN_TOKEN = "fb403452340dc08f8f7f6d026129130813d7afe6"
OPENWEATHER_TOKEN = "cb8b4a1479a995fa002f5ce5982e6b23"
CITY = "karachi"
CITY_COORDS = {"lat": 24.8607, "lon": 67.0011}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KarachiAQICollector:
    def __init__(self):
        self.city = CITY
        self.aqi_token = AQICN_TOKEN
        self.weather_token = OPENWEATHER_TOKEN
    
    def get_current_aqi(self):
        """Get current AQI data for Karachi"""
        url = f"https://api.waqi.info/feed/{self.city}/?token={self.aqi_token}"
        
        try:
            response = requests.get(url, timeout=10)
            logger.info(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'ok':
                    return self._parse_aqi_data(data['data'])
                else:
                    logger.error(f"API Error: {data.get('data')}")
                    return None
            else:
                logger.error(f"HTTP Error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching AQI data: {e}")
            return None
    
    def _parse_aqi_data(self, data):
        """Parse AQI API response for Karachi"""
        parsed_data = {
            'city': self.city,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'aqi': data.get('aqi', -1),
            'dominant_pollutant': data.get('dominentpol', 'unknown'),
            
            # Pollutant levels
            'pm25': data.get('iaqi', {}).get('pm25', {}).get('v', None),
            'pm10': data.get('iaqi', {}).get('pm10', {}).get('v', None),
            'o3': data.get('iaqi', {}).get('o3', {}).get('v', None),
            'no2': data.get('iaqi', {}).get('no2', {}).get('v', None),
            'so2': data.get('iaqi', {}).get('so2', {}).get('v', None),
            'co': data.get('iaqi', {}).get('co', {}).get('v', None),
            
            # Weather data from AQI
            'temperature': data.get('iaqi', {}).get('t', {}).get('v', None),
            'humidity': data.get('iaqi', {}).get('h', {}).get('v', None),
            'pressure': data.get('iaqi', {}).get('p', {}).get('v', None),
            'wind_speed': data.get('iaqi', {}).get('w', {}).get('v', None),
        }
        
        # Enhance with OpenWeather data
        weather_data = self._get_weather_data()
        if weather_data:
            parsed_data.update(weather_data)
        
        logger.info(f"Collected AQI data: {parsed_data['aqi']}")
        return parsed_data
    
    def _get_weather_data(self):
        """Get additional weather data from OpenWeather"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={CITY_COORDS['lat']}&lon={CITY_COORDS['lon']}&appid={self.weather_token}&units=metric"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temp_ow': data['main']['temp'],
                    'humidity_ow': data['main']['humidity'],
                    'pressure_ow': data['main']['pressure'],
                    'wind_speed_ow': data['wind']['speed'],
                    'weather_description': data['weather'][0]['description']
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

if __name__ == "__main__":
    collector = KarachiAQICollector()
    current_data = collector.get_current_aqi()
    if current_data:
        print("✅ Current AQI Data for Karachi:")
        for key, value in current_data.items():
            print(f"   {key}: {value}")