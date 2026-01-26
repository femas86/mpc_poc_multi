"""Data schemas for Italian weather service."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class LocationInfo(BaseModel):
    """Geographic location information."""
    
    name: str = Field(..., description="Location name")
    country: Optional[str] = Field(..., description="Country code")
    admin1: Optional[str] = Field(None, description="Administrative area level 1 (region)")
    admin2: Optional[str] = Field(None, description="Administrative area level 2 (province)")
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    elevation: Optional[float] = Field(None, description="Elevation in meters")
    timezone: Optional[str] = Field(None, description="Timezone")
    population: Optional[int] = Field(None, description="Population")


class WeatherCurrent(BaseModel):
    """Current weather conditions."""
    
    temperature: float = Field(..., description="Temperature in Celsius")
    apparent_temperature: float = Field(..., description="Feels-like temperature")
    humidity: int = Field(..., description="Relative humidity percentage")
    precipitation: float = Field(..., description="Precipitation in mm")
    rain: float = Field(..., description="Rain in mm")
    weather_code: int = Field(..., description="WMO weather code")
    cloud_cover: int = Field(..., description="Cloud cover percentage")
    wind_speed: float = Field(..., description="Wind speed in km/h")
    wind_direction: int = Field(..., description="Wind direction in degrees")
    pressure: float = Field(..., description="Sea level pressure in hPa")
    time: datetime = Field(..., description="Observation time")


class HourlyForecast(BaseModel):
    """Hourly forecast data."""
    
    time: datetime
    temperature: float
    apparent_temperature: float
    precipitation_probability: int = Field(..., ge=0, le=100)
    precipitation: float
    rain: float
    weather_code: int
    cloud_cover: int
    wind_speed: float
    wind_direction: int
    humidity: int


class DailyForecast(BaseModel):
    """Daily forecast data."""
    
    date: datetime
    temperature_max: float
    temperature_min: float
    apparent_temperature_max: float
    apparent_temperature_min: float
    sunrise: datetime
    sunset: datetime
    precipitation_sum: float
    rain_sum: float
    precipitation_hours: float
    precipitation_probability_max: int = Field(..., ge=0, le=100)
    wind_speed_max: float
    wind_gusts_max: float
    wind_direction_dominant: int
    weather_code: int
    uv_index_max: float


class WeatherForecast(BaseModel):
    """Complete weather forecast."""
    
    location: LocationInfo
    current: WeatherCurrent
    hourly: list[HourlyForecast] = Field(default_factory=list)
    daily: list[DailyForecast] = Field(default_factory=list)
    timezone: str
    elevation: float


class WeatherAlert(BaseModel):
    """Weather alert/warning."""
    
    event: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Severity level")
    headline: str = Field(..., description="Alert headline")
    description: str = Field(..., description="Detailed description")
    start_time: datetime
    end_time: datetime
    affected_areas: list[str] = Field(default_factory=list)
