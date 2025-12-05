"""Weather Italy MCP Server - Open-Meteo API with geocoding."""

from .server_it import WeatherItalyServer
from .wita_api_client import OpenMeteoClient, GeocodingClient
from .wita_schemas import (
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