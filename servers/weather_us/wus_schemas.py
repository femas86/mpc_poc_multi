import datetime
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


"""Data schemas for US National Weather Service."""
class WeatherQuery(BaseModel):
    city: str = Field(description= "Nome città degli USA")
    state: str = Field(description= "Codice stato (es: CA, NY)", min_length=2, max_length=2)

class AlertQuery(BaseModel):
    state: str = Field(description= "Codice stato (es: CA, NY)", min_length=2, max_length=2)

# class WeatherCurrent(BaseModel):
#     temperature: float = Field(description="Temperature in Fahrenheit")
#     windSpeed: float = Field(description="Wind speed in mph")
#     windDirection: str = Field(description="Wind direction")
#     condition: str  = Field(description="Weather condition description")
#     humidity: int = Field(description="Relative humidity percentage")
#     detailedForecast: str = Field(description="Detailed forecast description")
#     location: str = Field(description="Location name")

# class WeatherAlert(BaseModel):
#     """Weather alert/warning."""
#     event: str = Field(..., description="Alert type")
#     severity: str = Field(..., description="Severity level")
#     headline: str = Field(..., description="Alert headline")
#     description: str = Field(..., description="Detailed description")
#     instruction: Optional[str] = Field(None, description="Safety instructions")
#     start_time: datetime
#     end_time: datetime
#     affected_areas: list[str] = Field(default_factory=list)

class GridPoint(BaseModel):
    """NWS Grid point information."""
    
    office: str = Field(..., description="Weather Forecast Office")
    grid_x: int = Field(..., description="Grid X coordinate")
    grid_y: int = Field(..., description="Grid Y coordinate")
    forecast_url: str = Field(..., description="Forecast URL")
    forecast_hourly_url: str = Field(..., description="Hourly forecast URL")
    observation_stations_url: str = Field(..., description="Observation stations URL")


class ObservationStation(BaseModel):
    """Weather observation station."""
    
    station_id: str = Field(..., description="Station identifier")
    name: str = Field(..., description="Station name")
    timezone: str = Field(..., description="Timezone")
    forecast_office: str = Field(..., description="Forecast office")


class WeatherPeriod(BaseModel):
    """Weather forecast period."""
    
    number: int = Field(..., description="Period number")
    name: str = Field(..., description="Period name (e.g., 'Tonight', 'Wednesday')")
    start_time: datetime = Field(..., description="Period start time")
    end_time: datetime = Field(..., description="Period end time")
    is_daytime: bool = Field(..., description="Is daytime period")
    temperature: int = Field(..., description="Temperature in Fahrenheit")
    temperature_unit: str = Field(default="F", description="Temperature unit")
    temperature_trend: Optional[str] = Field(None, description="Temperature trend")
    wind_speed: str = Field(..., description="Wind speed (e.g., '10 mph')")
    wind_direction: str = Field(..., description="Wind direction (e.g., 'NW')")
    icon: str = Field(..., description="Weather icon URL")
    short_forecast: str = Field(..., description="Short forecast description")
    detailed_forecast: str = Field(..., description="Detailed forecast description")
    precipitation_probability: Optional[int] = Field(None, description="Precipitation probability")


class WeatherForecastUSA(BaseModel):
    """Complete weather forecast from NWS."""
    
    location: str = Field(..., description="Location description")
    latitude: float
    longitude: float
    grid_point: GridPoint
    updated: datetime = Field(..., description="Forecast update time")
    periods: list[WeatherPeriod] = Field(default_factory=list)
    elevation_feet: Optional[float] = Field(None, description="Elevation in feet")


class WeatherAlert(BaseModel):
    """NWS Weather Alert."""
    
    alert_id: str = Field(..., description="Alert identifier")
    area_desc: str = Field(..., description="Affected area description")
    event: str = Field(..., description="Event type")
    severity: str = Field(..., description="Severity level")
    certainty: str = Field(..., description="Certainty level")
    urgency: str = Field(..., description="Urgency level")
    headline: str = Field(..., description="Alert headline")
    description: str = Field(..., description="Alert description")
    instruction: Optional[str] = Field(None, description="Instructions")
    effective: datetime = Field(..., description="Effective time")
    expires: datetime = Field(..., description="Expiration time")
    sender_name: str = Field(..., description="Issuing office")


class CurrentObservation(BaseModel):
    """Current weather observation."""
    
    station_id: str
    timestamp: datetime
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    dewpoint: Optional[float] = Field(None, description="Dewpoint in Celsius")
    wind_direction: Optional[int] = Field(None, description="Wind direction in degrees")
    wind_speed: Optional[float] = Field(None, description="Wind speed in km/h")
    wind_gust: Optional[float] = Field(None, description="Wind gust in km/h")
    barometric_pressure: Optional[float] = Field(None, description="Pressure in Pascals")
    sea_level_pressure: Optional[float] = Field(None, description="Sea level pressure in Pascals")
    visibility: Optional[float] = Field(None, description="Visibility in meters")
    relative_humidity: Optional[float] = Field(None, description="Relative humidity percentage")
    wind_chill: Optional[float] = Field(None, description="Wind chill in Celsius")
    heat_index: Optional[float] = Field(None, description="Heat index in Celsius")
    text_description: Optional[str] = Field(None, description="Text description")