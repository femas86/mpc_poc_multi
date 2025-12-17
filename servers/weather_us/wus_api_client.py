"""API clients for Weather.gov (National Weather Service)."""

import httpx
import asyncio
from functools import lru_cache
from typing import Optional, Dict
from datetime import datetime
from config.settings import Settings, get_settings
from shared.logging_config import get_logger
from shared.utils import retry_async
from servers.weather_us.wus_schemas import (
    GridPoint,
    WeatherForecastUSA,
    WeatherPeriod,
    WeatherAlert,
    CurrentObservation,
    ObservationStation,
)

logger = get_logger(__name__)
settings= Settings()
# print(settings.weathergov_user_agent)

class GeoCodingClient:

    def __init__(self):
        self.headers = {
            'User-Agent': settings.weathergov_user_agent,
            'Accept': 'application/json'
        }
        self.client = httpx.AsyncClient(headers=self.headers)

    GEOCODING_URL = "https://nominatim.openstreetmap.org/search"

    @lru_cache(maxsize=256)
    @retry_async(max_attempts=3, delay=1.0)
    async def geocode(self, city: str, state: str) -> Dict:
        """Converti città/stato in coordinate"""
        # Usa servizio geocoding di openstreetmap.org
        # Use a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests

        async with semaphore:
            logger.debug("geocoding_request", city=city, state=state)
            response = await self.client.get(
                self.GEOCODING_URL,
                headers=self.headers,
                params={
                    'city': city,
                    'state': state,
                    'country': 'USA',
                    'format': 'json'
                }
            )
            response.raise_for_status()
            results = response.json()
            if results:
                logger.info("geocoding_success", city=city, state=state)
                return {
                    'lat': float(results[0]['lat']),
                    'lon': float(results[0]['lon'])
                }
            raise ValueError(f"Location not found: {city}, {state}")


    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class WeatherGovClient:
    """Client per API Weather.gov (NOAA)"""
    
    BASE_URL = "https://api.weather.gov"
    
    def __init__(self, user_agent: Optional[str] = None, timeout: Optional[int] = None):
        self.user_agent = user_agent or settings.weathergov_user_agent
        self.BASE_URL = settings.weathergov_base_url or self.BASE_URL
        self.timeout = timeout or settings.weathergov_timeout
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json'
        }
        self.client = httpx.AsyncClient(
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True,
        )
    
    @lru_cache(maxsize=256)
    @retry_async(max_attempts=3, delay=2.0)
    async def get_grid_point(self, latitude: float, longitude: float) -> GridPoint:
        """
        Get NWS grid point for coordinates.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            GridPoint information
            
        Raises:
            RuntimeError: If grid point lookup fails
        """
        try:
            url = f"{self.BASE_URL}/points/{latitude},{longitude}"
            
            logger.debug("grid_point_request", latitude=latitude, longitude=longitude)
            
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            
            properties = data["properties"]
            
            grid_point = GridPoint(
                office=properties["gridId"],
                grid_x=properties["gridX"],
                grid_y=properties["gridY"],
                forecast_url=properties["forecast"],
                forecast_hourly_url=properties["forecastHourly"],
                observation_stations_url=properties["observationStations"],
            )
            
            logger.info("grid_point_success", office=grid_point.office)
            return grid_point
            
        except httpx.HTTPError as e:
            logger.error("grid_point_error", error=str(e))
            raise RuntimeError(f"Grid point lookup failed: {e}") from e
    
    @lru_cache(maxsize=256) 
    @retry_async(max_attempts=3, delay=2.0)
    async def get_forecast(self, grid_point: GridPoint) -> WeatherForecastUSA:
        """
        Get weather forecast for grid point.
        
        Args:
            grid_point: NWS grid point
            
        Returns:
            Weather forecast
        """
        try:
            logger.debug("forecast_request", url=grid_point.forecast_url)
            
            response = await self.client.get(grid_point.forecast_url)
            response.raise_for_status()
            data = response.json()
            
            properties = data["properties"]
            
            # Parse periods
            periods = []
            for period_data in properties["periods"]:
                period = WeatherPeriod(
                    number=period_data["number"],
                    name=period_data["name"],
                    start_time=datetime.fromisoformat(period_data["startTime"]),
                    end_time=datetime.fromisoformat(period_data["endTime"]),
                    is_daytime=period_data["isDaytime"],
                    temperature=period_data["temperature"],
                    temperature_unit=period_data["temperatureUnit"],
                    temperature_trend=period_data.get("temperatureTrend"),
                    wind_speed=period_data["windSpeed"],
                    wind_direction=period_data["windDirection"],
                    icon=period_data["icon"],
                    short_forecast=period_data["shortForecast"],
                    detailed_forecast=period_data["detailedForecast"],
                    precipitation_probability=period_data.get("probabilityOfPrecipitation", {}).get("value"),
                )
                periods.append(period)
            
            # Get geometry for location info
            geometry = data["geometry"]
            coords = geometry["coordinates"][0]
            
            forecast = WeatherForecastUSA(
                location=properties.get("name", "Unknown"),
                latitude=coords[1],
                longitude=coords[0],
                grid_point=grid_point,
                updated=datetime.fromisoformat(properties["updated"]),
                periods=periods,
                elevation_feet=properties.get("elevation", {}).get("value"),
            )
            
            logger.info("forecast_success", periods=len(periods))
            return forecast
            
        except httpx.HTTPError as e:
            logger.error("forecast_error", error=str(e))
            raise RuntimeError(f"Forecast retrieval failed: {e}") from e
    
    @lru_cache(maxsize=256)
    @retry_async(max_attempts=3, delay=2.0)
    async def get_hourly_forecast(self, grid_point: GridPoint) -> list[WeatherPeriod]:
        """
        Get hourly forecast for grid point.
        
        Args:
            grid_point: NWS grid point
            
        Returns:
            List of hourly forecast periods
        """
        try:
            response = await self.client.get(grid_point.forecast_hourly_url)
            response.raise_for_status()
            data = response.json()
            
            periods = []
            for period_data in data["properties"]["periods"]:
                period = WeatherPeriod(
                    number=period_data["number"],
                    name=period_data["name"],
                    start_time=datetime.fromisoformat(period_data["startTime"]),
                    end_time=datetime.fromisoformat(period_data["endTime"]),
                    is_daytime=period_data["isDaytime"],
                    temperature=period_data["temperature"],
                    temperature_unit=period_data["temperatureUnit"],
                    temperature_trend=period_data.get("temperatureTrend"),
                    wind_speed=period_data["windSpeed"],
                    wind_direction=period_data["windDirection"],
                    icon=period_data["icon"],
                    short_forecast=period_data["shortForecast"],
                    detailed_forecast=period_data["detailedForecast"],
                    precipitation_probability=period_data.get("probabilityOfPrecipitation", {}).get("value"),
                )
                periods.append(period)
            
            logger.info("hourly_forecast_success", periods=len(periods))
            return periods
            
        except httpx.HTTPError as e:
            logger.error("hourly_forecast_error", error=str(e))
            raise RuntimeError(f"Hourly forecast retrieval failed: {e}") from e
        
    # @lru_cache(maxsize=256)
    # async def get_point(self, lat: float, lon: float) -> Dict:
    #     """Ottieni metadata punto griglia"""
    #     response = await self.client.get(
    #         f"{self.BASE_URL}/points/{lat},{lon}"
    #     )
    #     response.raise_for_status()
    #     return response.json()
    
    # async def get_forecast(self, forecast_url: str) -> Dict:
    #     """Ottieni forecast da URL"""
    #     response = await self.client.get(forecast_url)
    #     response.raise_for_status()
    #     return response.json()

    
    @retry_async(max_attempts=3, delay=2.0)
    async def get_alerts(self, state: str, latitude: float, longitude: float) -> list[WeatherAlert]:
        """
        Get active weather alerts for location.
        
        Args:
            state: State code (e.g., 'CA'), alternative to coordinates
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            List of active alerts
        """
        try:
            if state:
                url = f"{self.BASE_URL}/alerts/active/zone/{state}"
                logger.debug("alerts_request", state = state)
                response = await self.client.get(url, self.headers)
                response.raise_for_status()
                if not response or  "features" not in response.json():
                    logger.info("no_alerts_found", state=state)
                    return ["Unable to fetch alerts or no alerts found."]
            else:       
                url = f"{self.BASE_URL}/alerts/active"
                params = {"point": f"{latitude},{longitude}"}
                logger.debug("alerts_request", latitude=latitude, longitude=longitude)
            
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                if not response or  "features" not in response.json():
                    logger.info("no_alerts_found", state=state)
                    return ["Unable to fetch alerts or no alerts found."]
            
            data = response.json()
            alerts = []
            for feature in data.get("features", []):
                props = feature["properties"]
                
                alert = WeatherAlert(
                    alert_id=props["id"],
                    area_desc=props["areaDesc"],
                    event=props["event"],
                    severity=props["severity"],
                    certainty=props["certainty"],
                    urgency=props["urgency"],
                    headline=props["headline"],
                    description=props["description"],
                    instruction=props.get("instruction"),
                    effective=datetime.fromisoformat(props["effective"]),
                    expires=datetime.fromisoformat(props["expires"]),
                    sender_name=props["senderName"],
                )
                alerts.append(alert)
            
            logger.info("alerts_retrieved", count=len(alerts))
            return alerts
            
        except httpx.HTTPError as e:
            logger.error("alerts_error", error=str(e))
            return []
    
    @retry_async(max_attempts=3, delay=2.0)
    async def get_observation_stations(self, grid_point: GridPoint) -> list[ObservationStation]:
        """
        Get observation stations for grid point.
        
        Args:
            grid_point: NWS grid point
            
        Returns:
            List of observation stations
        """
        try:
            response = await self.client.get(grid_point.observation_stations_url)
            response.raise_for_status()
            data = response.json()
            
            stations = []
            for feature in data.get("features", []):
                props = feature["properties"]
                
                station = ObservationStation(
                    station_id=props["stationIdentifier"],
                    name=props["name"],
                    timezone=props["timeZone"],
                    forecast_office=props.get("forecast", ""),
                )
                stations.append(station)
            
            logger.info("stations_retrieved", count=len(stations))
            return stations
            
        except httpx.HTTPError as e:
            logger.error("stations_error", error=str(e))
            return []
    
    @retry_async(max_attempts=3, delay=2.0)
    async def get_latest_observation(self, station_id: str) -> Optional[CurrentObservation]:
        """
        Get latest observation from station.
        
        Args:
            station_id: Station identifier
            
        Returns:
            Current observation if available
        """
        try:
            url = f"{self.BASE_URL}/stations/{station_id}/observations/latest"
            
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            
            props = data["properties"]
            
            observation = CurrentObservation(
                station_id=station_id,
                timestamp=datetime.fromisoformat(props["timestamp"]),
                temperature=props.get("temperature", {}).get("value"),
                dewpoint=props.get("dewpoint", {}).get("value"),
                wind_direction=props.get("windDirection", {}).get("value"),
                wind_speed=props.get("windSpeed", {}).get("value"),
                wind_gust=props.get("windGust", {}).get("value"),
                barometric_pressure=props.get("barometricPressure", {}).get("value"),
                sea_level_pressure=props.get("seaLevelPressure", {}).get("value"),
                visibility=props.get("visibility", {}).get("value"),
                relative_humidity=props.get("relativeHumidity", {}).get("value"),
                wind_chill=props.get("windChill", {}).get("value"),
                heat_index=props.get("heatIndex", {}).get("value"),
                text_description=props.get("textDescription"),
            )
            
            logger.info("observation_retrieved", station=station_id)
            return observation
            
        except httpx.HTTPError as e:
            logger.error("observation_error", error=str(e), station=station_id)
            return None
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
 
    

