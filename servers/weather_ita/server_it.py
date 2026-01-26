"""MCP Server for Italian weather using Open-Meteo API."""
#import shared
from typing import Any
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from shared.logging_config import get_logger
from servers.weather_ita.wita_api_client import GeocodingClient, OpenMeteoClient
from servers.weather_ita.wita_schemas import WeatherForecast, LocationInfo

logger = get_logger(__name__)

# Create MCP server
mcp = FastMCP(
    name="WeatherItaly",
    instructions="Provides weather forecasts for Italian cities using Open-Meteo API with geocoding support"
)

# Initialize clients
geocoding_client = GeocodingClient()
weather_client = OpenMeteoClient()


@mcp.tool()
async def search_italian_city(city_name: str, ctx: Context[ServerSession, None]) -> dict[str, Any]:
    """
    MANDATORY FIRST STEP for any weather request in Italy.
    Use this to validate the city name and obtain its official coordinates (latitude/longitude).
    Args:
        city_name: The name of the Italian city to find (e.g., "Firenze").  
    Returns:
        Exact location data including coordinates. REQUIRED before calling get_weather_italy.
    """
    await ctx.info(f"Searching for Italian city: {city_name}")
    try:
        locations = await geocoding_client.search_location(city_name, count=5)
        # Filter for Italian locations
        italian_locations = [loc for loc in locations if loc.country == "IT"]
        if not italian_locations:
            return {
                "found": False,
                "message": f"No Italian cities found matching '{city_name}'",
                "results": [],
            }
        
        results = []
        results = [LocationInfo(
            name=loc.name,
            region=loc.admin1,
            province=loc.admin2,
            latitude=loc.latitude,
            longitude=loc.longitude,
            population=loc.population,
        ).model_dump() for loc in italian_locations]
        
        await ctx.info(f"Found {len(results)} Italian locations")
        
        return {
            "found": True,
            "count": len(results),
            "results": results,
        }
        
    except Exception as e:
        await ctx.error(f"Search failed: {str(e)}")
        return {"found": False, "error": str(e)}


@mcp.tool()
async def get_weather_italy(
    latitude: float,
    longitude: float,
    city_name: str, # Opzionale, solo per log/display
    forecast_days: int = 7,
    include_hourly: bool = False,
    ctx: Context[ServerSession, None] = None,
) -> dict[str, Any]:
    """
    Get real-time weather forecast for an Italian location using coordinates.
    DO NOT guess coordinates. Get them from 'search_italian_city' first.
    Args:
        latitude: Latitude obtained from search_italian_city.
        longitude: Longitude obtained from search_italian_city.
        city_name: Name of the city (for display purposes).
        forecast_days: Number of days to forecast (1-16).
    Returns:
        Complete weather forecast with current conditions and daily/hourly forecasts
    """
    await ctx.info(f"Getting weather for {city_name}, Italy")
    
    try:
        # Get weather forecast
        await ctx.debug("Fetching weather data...")
        weather_data = await weather_client.get_forecast(
            latitude=latitude,
            longitude=longitude,
            forecast_days=forecast_days,
            include_hourly=include_hourly,
        )
        
        # Parse forecast
        forecast = weather_client.parse_forecast(weather_data, LocationInfo(name=city_name, latitude=latitude, longitude=longitude))
        
        await ctx.info(f"Weather forecast retrieved for {city_name}")
        return forecast.model_dump(exclude_none=True)
        
    except Exception as e:
        await ctx.error(f"Weather retrieval failed: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.resource("weather://italy/current/{city}")
async def current_weather_resource(city: str) -> str:
    """
    Get instant current weather in Italian city.
    
    Args:
        city: Italian city name
        
    Returns:
        Formatted current weather string
    """
    try:
        locations = await geocoding_client.search_location(city, count=1)
        italian_locs = [loc for loc in locations if loc.country == "IT"]
        
        if not italian_locs:
            return f"City '{city}' not found in Italy"
        
        location = italian_locs[0]
        weather_data = await weather_client.get_forecast(
            latitude=location.latitude,
            longitude=location.longitude,
            forecast_days=1,
            include_hourly=False,
        )
        
        forecast = weather_client.parse_forecast(weather_data, location)
        current = forecast.current
        
        weather_desc = weather_client.get_weather_description(current.weather_code)
        
        return f"""Current Weather for {location.name}, {location.admin1}, Italy
                Time: {current.time.strftime('%Y-%m-%d %H:%M')}
                Temperature: {current.temperature}°C (feels like {current.apparent_temperature}°C)
                Weather: {weather_desc}
                Humidity: {current.humidity}%
                Wind: {current.wind_speed} km/h from {current.wind_direction}°
                Pressure: {current.pressure} hPa
                Cloud Cover: {current.cloud_cover}%
                """
  
    except Exception as e:
        logger.error("current_weather_resource_error", error=str(e), city=city)
        return f"Error retrieving weather for {city}: {str(e)}"

class WeatherItalyServer:
    """Wrapper class for Weather Italy MCP Server."""
    
    def __init__(self):
        """Initialize Weather Italy server."""
        self.mcp = mcp
        self.geocoding_client = geocoding_client
        self.weather_client = weather_client
        logger.info("weather_italy_server_initialized")
    
    async def start(self):
        """Start the server."""
        logger.info("weather_italy_server_started")
    
    async def stop(self):
        """Stop the server and cleanup."""
        await self.geocoding_client.close()
        await self.weather_client.close()
        logger.info("weather_italy_server_stopped")
    
    def get_mcp_server(self) -> FastMCP:
        """Get the FastMCP server instance."""
        return self.mcp
    
if __name__ == "__main__":
    """
    Run server in stdio mode for MCP host connection.
    
    This allows the MCP host to:
    1. Connect via stdio
    2. Discover tools with list_tools()
    3. Discover resources with list_resources()
    4. Call tools and read resources
    """
    import asyncio
    from shared.logging_config import setup_logging
    
    # 1. Initialize the logging system to use stderr
    setup_logging()
    
    async def run_server():
        """Run the MCP server in stdio mode."""
        server = WeatherItalyServer()
        await server.start()
        
        # Run FastMCP server in stdio mode
        # This is what the MCP host connects to
        await mcp.run_stdio_async()
        
        await server.stop()
    
    # Run the server
    asyncio.run(run_server())
