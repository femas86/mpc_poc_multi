"""API clients for Open-Meteo and Geocoding services."""

import httpx
from typing import Optional
from datetime import datetime, timedelta
from config.settings import get_settings
from shared.logging_config import get_logger
from shared.utils import retry_async
from servers.weather_ita.wita_schemas import (
    LocationInfo,
    WeatherCurrent,
    WeatherForecast,
    HourlyForecast,
    DailyForecast,
)

logger = get_logger(__name__)


class GeocodingClient:
    """Client for Open-Meteo Geocoding API."""
    
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    
    def __init__(self, timeout: Optional[int] = None):
        settings = get_settings()
        self.timeout = timeout or settings.openmeteo_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    @retry_async(max_attempts=3, delay=1.0)
    async def search_location(
        self, query: str, count: int = 5, language: str = "it"
    ) -> list[LocationInfo]:
        """
        Search for locations by name.
        
        Args:
            query: Location name to search
            count: Maximum number of results
            language: Language for results
            
        Returns:
            List of LocationInfo objects
        """
        try:
            params = {
                "name": query,
                "count": count,
                "language": language,
                "format": "json",
            }
            
            logger.debug("geocoding_request", query=query, count=count)
            
            response = await self.client.get(self.GEOCODING_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "results" not in data:
                logger.warning("geocoding_no_results", query=query)
                return []
            
            locations = []
            for result in data["results"]:
                location = LocationInfo(
                    name=result.get("name", ""),
                    country=result.get("country_code", ""),
                    admin1=result.get("admin1"),
                    admin2=result.get("admin2"),
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                    elevation=result.get("elevation"),
                    timezone=result.get("timezone"),
                    population=result.get("population"),
                )
                locations.append(location)
            
            logger.info("geocoding_success", query=query, count=len(locations))
            return locations
            
        except httpx.HTTPError as e:
            logger.error("geocoding_error", error=str(e), query=query)
            raise RuntimeError(f"Geocoding failed: {e}") from e
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class OpenMeteoClient:
    """Client for Open-Meteo Weather API."""
    
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    # WMO Weather codes mapping
    WEATHER_CODES = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    
    def __init__(self, timeout: Optional[int] = None):
        settings = get_settings()
        self.timeout = timeout or settings.openmeteo_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    @retry_async(max_attempts=3, delay=1.0)
    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7,
        include_hourly: bool = True,
    ) -> dict:
        """
        Get weather forecast for location.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            forecast_days: Number of days to forecast (1-16)
            include_hourly: Include hourly forecast
            
        Returns:
            Weather forecast data
        """
        try:
            # Current weather variables
            current_vars = [
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "weather_code",
                "cloud_cover",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
            
            # Daily variables
            daily_vars = [
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "sunrise",
                "sunset",
                "precipitation_sum",
                "rain_sum",
                "precipitation_hours",
                "precipitation_probability_max",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "wind_direction_10m_dominant",
                "weather_code",
                "uv_index_max",
            ]
            
            # Hourly variables
            hourly_vars = [
                "temperature_2m",
                "apparent_temperature",
                "precipitation_probability",
                "precipitation",
                "rain",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "relative_humidity_2m",
            ]
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(current_vars),
                "daily": ",".join(daily_vars),
                "timezone": "auto",
                "forecast_days": min(forecast_days, 16),
            }
            
            if include_hourly:
                params["hourly"] = ",".join(hourly_vars)
            
            logger.debug(
                "weather_forecast_request",
                latitude=latitude,
                longitude=longitude,
                days=forecast_days,
            )
            
            response = await self.client.get(self.FORECAST_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            logger.info("weather_forecast_success", latitude=latitude, longitude=longitude)
            return data
            
        except httpx.HTTPError as e:
            logger.error("weather_forecast_error", error=str(e))
            raise RuntimeError(f"Weather forecast failed: {e}") from e
    
    def parse_forecast(self, data: dict, location: LocationInfo) -> WeatherForecast:
        """Parse API response into WeatherForecast."""
        
        # Parse current weather
        current_data = data["current"]
        current = WeatherCurrent(
            temperature=current_data["temperature_2m"],
            apparent_temperature=current_data["apparent_temperature"],
            humidity=current_data["relative_humidity_2m"],
            precipitation=current_data["precipitation"],
            rain=current_data["rain"],
            weather_code=current_data["weather_code"],
            cloud_cover=current_data["cloud_cover"],
            wind_speed=current_data["wind_speed_10m"],
            wind_direction=current_data["wind_direction_10m"],
            pressure=current_data["pressure_msl"],
            time=datetime.fromisoformat(current_data["time"]),
        )
        
        # Parse daily forecast
        daily_forecasts = []
        daily_data = data["daily"]
        for i in range(len(daily_data["time"])):
            daily = DailyForecast(
                date=datetime.fromisoformat(daily_data["time"][i]),
                temperature_max=daily_data["temperature_2m_max"][i],
                temperature_min=daily_data["temperature_2m_min"][i],
                apparent_temperature_max=daily_data["apparent_temperature_max"][i],
                apparent_temperature_min=daily_data["apparent_temperature_min"][i],
                sunrise=datetime.fromisoformat(daily_data["sunrise"][i]),
                sunset=datetime.fromisoformat(daily_data["sunset"][i]),
                precipitation_sum=daily_data["precipitation_sum"][i],
                rain_sum=daily_data["rain_sum"][i],
                precipitation_hours=daily_data["precipitation_hours"][i],
                precipitation_probability_max=daily_data["precipitation_probability_max"][i],
                wind_speed_max=daily_data["wind_speed_10m_max"][i],
                wind_gusts_max=daily_data["wind_gusts_10m_max"][i],
                wind_direction_dominant=daily_data["wind_direction_10m_dominant"][i],
                weather_code=daily_data["weather_code"][i],
                uv_index_max=daily_data["uv_index_max"][i],
            )
            daily_forecasts.append(daily)
        
        # Parse hourly forecast if available
        hourly_forecasts = []
        if "hourly" in data:
            hourly_data = data["hourly"]
            for i in range(len(hourly_data["time"])):
                hourly = HourlyForecast(
                    time=datetime.fromisoformat(hourly_data["time"][i]),
                    temperature=hourly_data["temperature_2m"][i],
                    apparent_temperature=hourly_data["apparent_temperature"][i],
                    precipitation_probability=hourly_data["precipitation_probability"][i],
                    precipitation=hourly_data["precipitation"][i],
                    rain=hourly_data["rain"][i],
                    weather_code=hourly_data["weather_code"][i],
                    cloud_cover=hourly_data["cloud_cover"][i],
                    wind_speed=hourly_data["wind_speed_10m"][i],
                    wind_direction=hourly_data["wind_direction_10m"][i],
                    humidity=hourly_data["relative_humidity_2m"][i],
                )
                hourly_forecasts.append(hourly)
        
        return WeatherForecast(
            location=location,
            current=current,
            hourly=hourly_forecasts,
            daily=daily_forecasts,
            timezone=data["timezone"],
            elevation=data["elevation"],
        )
    
    def get_weather_description(self, code: int) -> str:
        """Get human-readable weather description from WMO code."""
        return self.WEATHER_CODES.get(code, f"Unknown ({code})")
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
