from typing import Any, Optional
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from shared.logging_config import get_logger
from servers.weather_us.wus_api_client import WeatherGovClient, GeoCodingClient
from servers.weather_us.wus_schemas import WeatherForecastUSA, WeatherAlert, WeatherPeriod, GridPoint, ObservationStation

logger = get_logger(__name__)

# Create MCP server
mcp = FastMCP(
    name="WeatherUSA",
    instructions="Provides weather forecasts for US locations using National Weather Service API"
)

# Initialize clients
geocoding_client = GeoCodingClient()
weather_client = WeatherGovClient()


@mcp.tool()
async def search_us_location(location: str, ctx: Context[ServerSession, None]) -> dict[str, Any]:
    """
    Search for US location and get coordinates.
    
    Args:
        location: US location (city, state, ZIP code, or address)
        
    Returns:
        Location information with coordinates
    """
    await ctx.info(f"Searching for US location: {location}")
    
    try:
        result = await geocoding_client.geocode_address(location)
        
        if not result:
            return {
                "found": False,
                "message": f"Location '{location}' not found in USA",
            }
        
        await ctx.info(f"Found: {result['formatted_address']}")
        
        return {
            "found": True,
            "location": result["formatted_address"],
            "latitude": result["latitude"],
            "longitude": result["longitude"],
        }
        
    except Exception as e:
        await ctx.error(f"Location search failed: {str(e)}")
        return {"found": False, "error": str(e)}


@mcp.tool()
async def get_weather_usa(
    location: str,
    include_hourly: bool = False,
    include_alerts: bool = True,
    ctx: Context[ServerSession, None] = None,
) -> dict[str, Any]:
    """
    Get weather forecast for US location.
    
    Args:
        location: US location (city, state, ZIP, or address)
        include_hourly: Include hourly forecast
        include_alerts: Include weather alerts
        
    Returns:
        Complete weather forecast from National Weather Service
    """
    await ctx.info(f"Getting weather for {location}, USA")
    
    try:
        # Geocode location
        await ctx.debug("Geocoding location...")
        geo_result = await geocoding_client.geocode_address(location)
        
        if not geo_result:
            return {
                "success": False,
                "error": f"Location '{location}' not found in USA",
            }
        
        latitude = geo_result["latitude"]
        longitude = geo_result["longitude"]
        formatted_address = geo_result["formatted_address"]
        
        await ctx.info(f"Found: {formatted_address}")
        
        # Get grid point
        await ctx.debug("Getting NWS grid point...")
        grid_point = await weather_client.get_grid_point(latitude, longitude)
        
        # Get forecast
        await ctx.debug("Fetching forecast...")
        forecast = await weather_client.get_forecast(grid_point)
        
        # Prepare result
        result = {
            "success": True,
            "location": formatted_address,
            "latitude": latitude,
            "longitude": longitude,
            "office": grid_point.office,
            "updated": forecast.updated.isoformat(),
            "forecast": [
                {
                    "period": period.name,
                    "start_time": period.start_time.isoformat(),
                    "end_time": period.end_time.isoformat(),
                    "is_daytime": period.is_daytime,
                    "temperature": period.temperature,
                    "temperature_unit": period.temperature_unit,
                    "wind_speed": period.wind_speed,
                    "wind_direction": period.wind_direction,
                    "short_forecast": period.short_forecast,
                    "detailed_forecast": period.detailed_forecast,
                    "precipitation_probability": period.precipitation_probability,
                }
                for period in forecast.periods
            ],
        }
        
        # Add hourly forecast if requested
        if include_hourly:
            await ctx.debug("Fetching hourly forecast...")
            hourly_periods = await weather_client.get_hourly_forecast(grid_point)
            result["hourly_forecast"] = [
                {
                    "time": period.start_time.isoformat(),
                    "temperature": period.temperature,
                    "wind_speed": period.wind_speed,
                    "wind_direction": period.wind_direction,
                    "short_forecast": period.short_forecast,
                    "precipitation_probability": period.precipitation_probability,
                }
                for period in hourly_periods[:24]  # Next 24 hours
            ]
        
        # Add alerts if requested
        if include_alerts:
            await ctx.debug("Checking for weather alerts...")
            alerts = await weather_client.get_alerts(latitude, longitude)
            if alerts:
                result["alerts"] = [
                    {
                        "event": alert.event,
                        "severity": alert.severity,
                        "urgency": alert.urgency,
                        "headline": alert.headline,
                        "description": alert.description[:500],  # Truncate
                        "effective": alert.effective.isoformat(),
                        "expires": alert.expires.isoformat(),
                    }
                    for alert in alerts
                ]
                await ctx.warning(f"Found {len(alerts)} active weather alerts")
        
        await ctx.info(f"Weather forecast retrieved for {formatted_address}")
        return result
        
    except Exception as e:
        await ctx.error(f"Weather retrieval failed: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_current_conditions_usa(
    location: str,
    ctx: Context[ServerSession, None] = None,
) -> dict[str, Any]:
    """
    Get current weather conditions for US location.
    
    Args:
        location: US location
        
    Returns:
        Current weather observation
    """
    await ctx.info(f"Getting current conditions for {location}")
    
    try:
        # Geocode
        geo_result = await geocoding_client.geocode_address(location)
        if not geo_result:
            return {"success": False, "error": "Location not found"}
        
        latitude = geo_result["latitude"]
        longitude = geo_result["longitude"]
        
        # Get grid point and stations
        grid_point = await weather_client.get_grid_point(latitude, longitude)
        stations = await weather_client.get_observation_stations(grid_point)
        
        if not stations:
            return {"success": False, "error": "No observation stations found"}
        
        # Try to get observation from first station
        observation = await weather_client.get_latest_observation(stations[0].station_id)
        
        if not observation:
            return {"success": False, "error": "No current observation available"}
        
        # Convert Celsius to Fahrenheit for consistency with NWS forecasts
        def c_to_f(celsius: Optional[float]) -> Optional[float]:
            return (celsius * 9/5) + 32 if celsius is not None else None
        
        result = {
            "success": True,
            "location": geo_result["formatted_address"],
            "station": stations[0].name,
            "station_id": stations[0].station_id,
            "observation_time": observation.timestamp.isoformat(),
            "temperature_f": c_to_f(observation.temperature),
            "temperature_c": observation.temperature,
            "dewpoint_f": c_to_f(observation.dewpoint),
            "humidity": observation.relative_humidity,
            "wind_speed_mph": observation.wind_speed * 0.621371 if observation.wind_speed else None,
            "wind_direction": observation.wind_direction,
            "pressure_mb": observation.barometric_pressure / 100 if observation.barometric_pressure else None,
            "visibility_miles": observation.visibility * 0.000621371 if observation.visibility else None,
            "description": observation.text_description,
        }
        
        await ctx.info("Current conditions retrieved")
        return result
        
    except Exception as e:
        await ctx.error(f"Current conditions retrieval failed: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.resource("weather://usa/forecast/{location}")
async def forecast_resource(location: str) -> str:
    """
    Resource endpoint for US weather forecast.
    
    Args:
        location: US location
        
    Returns:
        Formatted forecast string
    """
    try:
        geo_result = await geocoding_client.geocode_address(location)
        if not geo_result:
            return f"Location '{location}' not found in USA"
        
        grid_point = await weather_client.get_grid_point(
            geo_result["latitude"],
            geo_result["longitude"]
        )
        
        forecast = await weather_client.get_forecast(grid_point)
        
        output = [f"Weather Forecast for {geo_result['formatted_address']}"]
        output.append(f"Updated: {forecast.updated.strftime('%Y-%m-%d %H:%M')}")
        output.append(f"Forecast Office: {grid_point.office}\n")
        
        for period in forecast.periods[:5]:  # Next 5 periods
            output.append(f"=== {period.name} ===")
            output.append(f"Temperature: {period.temperature}°{period.temperature_unit}")
            output.append(f"Wind: {period.wind_speed} {period.wind_direction}")
            if period.precipitation_probability:
                output.append(f"Precipitation: {period.precipitation_probability}%")
            output.append(f"{period.short_forecast}")
            output.append(f"{period.detailed_forecast}\n")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error("forecast_resource_error", error=str(e), location=location)
        return f"Error retrieving forecast for {location}: {str(e)}"


class WeatherUSAServer:
    """Wrapper class for Weather USA MCP Server."""
    
    def __init__(self):
        """Initialize Weather USA server."""
        self.mcp = mcp
        self.geocoding_client = geocoding_client
        self.weather_client = weather_client
        logger.info("weather_usa_server_initialized")
    
    async def start(self):
        """Start the server."""
        logger.info("weather_usa_server_started")
    
    async def stop(self):
        """Stop the server and cleanup."""
        await self.geocoding_client.close()
        await self.weather_client.close()
        logger.info("weather_usa_server_stopped")
    
    def get_mcp_server(self) -> FastMCP:
        """Get the FastMCP server instance."""
        return self.mcp

if __name__ == "__main__":
    """Run server in stdio mode for MCP host connection."""
    import asyncio
    
    async def run_server():
        """Run the MCP server in stdio mode."""
        server = WeatherUSAServer()
        await server.start()
        
        # Run FastMCP server in stdio mode
        await mcp.run(transport="stdio")
        
        await server.stop()
    
    asyncio.run(run_server())