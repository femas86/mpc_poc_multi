"""Weather Italy MCP Server - Open-Meteo API with geocoding."""

from servers.weather_ita.server_it import WeatherItalyServer
from servers.weather_ita.wita_api_client import OpenMeteoClient, GeocodingClient
from servers.weather_ita.wita_schemas import (
    WeatherForecast,
    WeatherCurrent,
    LocationInfo,
    WeatherAlert,
)

__all__ = [
    "WeatherItalyServer",
    "OpenMeteoClient",
    "GeocodingClient",
    "WeatherForecast",
    "WeatherCurrent",
    "LocationInfo",
    "WeatherAlert",
]