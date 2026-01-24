from matplotlib.figure import Figure
from networkx import MultiDiGraph
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
import sys
from datetime import datetime
import argparse
import pickle
import asyncio
from pathlib import Path
from hashlib import md5
from typing import cast
from geopandas import GeoDataFrame
import pickle
from shapely.geometry import Point, Polygon as ShapelyPolygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CacheError(Exception):
    """Raised when a cache operation fails."""
    pass

CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)


THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

CACHE_DIR = ".cache"

class CacheError(Exception):
    pass


def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}")


def cache_set(key: str, value):
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}")


def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    
    # Verify fonts exist
    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"WARNING: Font not found: {path}")
            return None
    
    return fonts

FONTS = load_fonts()

def generate_output_filename(city, theme_name, output_format):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_')
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes

def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    
    if not os.path.exists(theme_file):
        print(f"WARNING: Theme file '{theme_file}' not found. Using default feature_based theme.")
        # Fallback to embedded default theme
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A",
            "building_fill": "#808080",
            "building_edge": "#404040",
            "building_alpha": 0.85
        }

    with open(theme_file, 'r') as f:
        theme = json.load(f)
        # Add building color fallbacks if not present in theme
        if 'building_fill' not in theme:
            theme['building_fill'] = '#808080'
        if 'building_edge' not in theme:
            theme['building_edge'] = '#404040'
        if 'building_alpha' not in theme:
            theme['building_alpha'] = 0.85
        print(f"[OK] Loaded theme: {theme.get('name', theme_name)}")
        if 'description' in theme:
            print(f"  {theme['description']}")
        return theme

# Load theme (can be changed via command line or input)
THEME = dict[str, str]()  # Will be loaded later

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def get_edge_colors_by_type(G):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []
    
    for u, v, data in G.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')
        
        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign color based on road type
        if highway in ['motorway', 'motorway_link']:
            color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']:
            color = THEME['road_residential']
        else:
            color = THEME['road_default']
        
        edge_colors.append(color)
    
    return edge_colors

def get_edge_widths_by_type(G):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign width based on road importance
        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4
        
        edge_widths.append(width)
    
    return edge_widths

def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"[OK] Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}")

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError:
            # If an event loop is already running, try using it to complete the coroutine.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running event loop in the same thread; raise a clear error.
                raise RuntimeError("Geocoder returned a coroutine while an event loop is already running. Run this script in a synchronous environment.")
            location = loop.run_until_complete(location)
    
    if location:
        # Use getattr to safely access address (helps static analyzers)
        addr = getattr(location, "address", None)
        if addr:
            print(f"[OK] Found: {addr}")
        else:
            print("[OK] Found location (address not available)")
        print(f"[OK] Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")
    
def get_crop_limits(G_proj, center_lat_lon, fig, dist):
    """
    Crop inward to preserve aspect ratio while guaranteeing
    full coverage of the requested radius.
    """
    lat, lon = center_lat_lon

    # Project center point into graph CRS
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=G_proj.graph["crs"]
        )[0]
    )
    center_x, center_y = center.x, center.y

    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height

    # Start from the *requested* radius
    half_x = dist
    half_y = dist

    # Cut inward to match aspect
    if aspect > 1:  # landscape → reduce height
        half_y = half_x / aspect
    else:           # portrait → reduce width
        half_x = half_y * aspect

    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


def fetch_graph(point, dist) -> MultiDiGraph | None:
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None:
        print("[OK] Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all', truncate_by_edge=True)
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, G)
        except CacheError as e:
            print(e)
        return G
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None

def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"[OK] Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def fetch_buildings(point, dist) -> GeoDataFrame | None:
    """
    Fetch building footprints from OSM with height-related tags.
    """
    lat, lon = point
    cache_key = f"buildings_{lat}_{lon}_{dist}"
    cached = cache_get(cache_key)
    if cached is not None:
        print("[OK] Using cached building data")
        return cast(GeoDataFrame, cached)

    try:
        # Fetch buildings with height tags
        buildings = ox.features_from_point(
            point,
            tags={'building': True},
            dist=dist
        )
        time.sleep(0.3)
        try:
            cache_set(cache_key, buildings)
        except CacheError as e:
            print(e)
        return buildings
    except Exception as e:
        print(f"OSMnx error while fetching buildings: {e}")
        return None


def extract_building_height(row, default_height=12.0, height_per_level=3.5, max_height=200.0):
    """
    Extract building height from OSM tags.
    Priority: height tag > building:levels > default
    Heights are capped at max_height for visualization.
    """
    height_value = None

    # Try direct height tag (in meters)
    height = row.get('height')
    if height is not None:
        try:
            # Handle strings like "25 m" or "25"
            if isinstance(height, str):
                height = height.replace('m', '').strip()
            height_value = float(height)
        except (ValueError, TypeError):
            pass

    # Try building:levels tag
    if height_value is None:
        levels = row.get('building:levels')
        if levels is not None:
            try:
                if isinstance(levels, str):
                    levels = levels.strip()
                height_value = float(levels) * height_per_level
            except (ValueError, TypeError):
                pass

    # Default height based on building type
    if height_value is None:
        building_type = row.get('building', '')
        if isinstance(building_type, str):
            if building_type in ['skyscraper', 'tower']:
                height_value = 80.0
            elif building_type in ['apartments', 'office', 'commercial']:
                height_value = 25.0
            elif building_type in ['house', 'residential', 'detached']:
                height_value = 8.0
            elif building_type in ['garage', 'shed', 'hut']:
                height_value = 4.0
            else:
                height_value = default_height
        else:
            height_value = default_height

    # Cap at max_height for reasonable visualization
    return min(height_value, max_height)


def simplify_polygon(geom, tolerance=2.0):
    """
    Simplify polygon geometry to reduce vertex count for performance.
    """
    if geom is None:
        return None
    try:
        simplified = geom.simplify(tolerance, preserve_topology=True)
        return simplified
    except Exception:
        return geom


def polygon_to_3d_faces(polygon, height, base_z=0):
    """
    Convert a 2D polygon to 3D box faces for Poly3DCollection.
    Returns list of faces (bottom, top, and sides).
    """
    if polygon is None or polygon.is_empty:
        return []

    faces = []

    # Get exterior coordinates
    if hasattr(polygon, 'exterior'):
        coords = list(polygon.exterior.coords)
    else:
        return []

    if len(coords) < 4:  # Need at least 3 points + closing point
        return []

    # Bottom face (z = base_z)
    bottom = [(x, y, base_z) for x, y in coords[:-1]]
    faces.append(bottom)

    # Top face (z = base_z + height)
    top = [(x, y, base_z + height) for x, y in coords[:-1]]
    faces.append(top)

    # Side faces (walls)
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        side = [
            (x1, y1, base_z),
            (x2, y2, base_z),
            (x2, y2, base_z + height),
            (x1, y1, base_z + height)
        ]
        faces.append(side)

    return faces


def create_building_mesh(buildings_gdf, theme, max_buildings=5000, simplify_tolerance=2.0):
    """
    Build Poly3DCollection from building GeoDataFrame.
    """
    if buildings_gdf is None or buildings_gdf.empty:
        return None

    all_faces = []
    building_count = 0

    # Filter to polygon geometries only
    buildings_poly = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    for idx, row in buildings_poly.iterrows():
        if building_count >= max_buildings:
            print(f"  Reached max buildings limit ({max_buildings})")
            break

        geom = row.geometry
        height = extract_building_height(row)

        # Handle MultiPolygon
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                simplified = simplify_polygon(poly, simplify_tolerance)
                faces = polygon_to_3d_faces(simplified, height)
                all_faces.extend(faces)
                building_count += 1
                if building_count >= max_buildings:
                    break
        else:
            simplified = simplify_polygon(geom, simplify_tolerance)
            faces = polygon_to_3d_faces(simplified, height)
            all_faces.extend(faces)
            building_count += 1

    if not all_faces:
        return None

    print(f"  Created mesh for {building_count} buildings ({len(all_faces)} faces)")

    # Create Poly3DCollection
    mesh = Poly3DCollection(
        all_faces,
        facecolors=theme.get('building_fill', '#808080'),
        edgecolors=theme.get('building_edge', '#404040'),
        linewidths=0.2,
        alpha=theme.get('building_alpha', 0.85)
    )

    return mesh


def create_3d_poster(
    city, country, point, dist, output_file, output_format,
    width=12, height=16, country_label=None, name_label=None,
    elevation=30.0, azimuth=-60.0,
    show_roads=True, show_water=True, show_parks=True,
    max_buildings=5000, zoom=1.5
):
    """
    Create a 3D map poster with extruded buildings.

    Args:
        zoom: Zoom factor to fill the frame (1.0 = default, higher = more zoomed in)
    """
    print(f"\nGenerating 3D map for {city}, {country}...")

    # Progress bar for data fetching
    with tqdm(total=4, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        compensated_dist = dist * (max(height, width) / min(height, width)) / 4
        G = fetch_graph(point, compensated_dist) if show_roads else None
        pbar.update(1)

        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        water = fetch_features(point, compensated_dist, tags={'natural': 'water', 'waterway': 'riverbank'}, name='water') if show_water else None
        pbar.update(1)

        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(point, compensated_dist, tags={'leisure': 'park', 'landuse': 'grass'}, name='parks') if show_parks else None
        pbar.update(1)

        # 4. Fetch Buildings
        pbar.set_description("Downloading building data")
        buildings = fetch_buildings(point, compensated_dist)
        pbar.update(1)

    print("[OK] All data retrieved successfully!")

    # Setup 3D figure
    print("Rendering 3D map...")
    fig = plt.figure(figsize=(width, height), facecolor=THEME['bg'])

    # Create 3D axes that fills the figure (leave small margin for text at bottom)
    ax = fig.add_axes([0, 0.15, 1, 0.85], projection='3d', facecolor=THEME['bg'])

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Disable automatic margins/padding
    ax.set_proj_type('persp', focal_length=0.2)  # Adjust perspective for better fill

    # Project data to metric CRS
    if G is not None:
        G_proj = ox.project_graph(G)
        crs = G_proj.graph['crs']
    elif buildings is not None:
        # Use buildings to determine CRS
        buildings_proj = ox.projection.project_gdf(buildings)
        crs = buildings_proj.crs
    else:
        print("ERROR: No data to render")
        return

    # Get bounds from graph or estimate from point
    if G is not None:
        crop_xlim, crop_ylim = get_crop_limits(G_proj, point, fig, compensated_dist)
        xlim = crop_xlim
        ylim = crop_ylim
    else:
        # Estimate bounds from point
        center = ox.projection.project_geometry(
            Point(point[1], point[0]),
            crs="EPSG:4326",
            to_crs=crs
        )[0]
        half = compensated_dist
        xlim = (center.x - half, center.x + half)
        ylim = (center.y - half, center.y + half)

    # Layer 1: Water (as flat polygons at z=0)
    if water is not None and not water.empty:
        water_polys = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polys.empty:
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                if crs:
                    water_polys = water_polys.to_crs(crs)

            for geom in water_polys.geometry:
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax.plot(x, y, zs=0, zdir='z', color=THEME['water'], linewidth=0.5)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, zs=0, zdir='z', color=THEME['water'], linewidth=0.5)

    # Layer 2: Parks (as flat polygons at z=0)
    if parks is not None and not parks.empty:
        parks_polys = parks[parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not parks_polys.empty:
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                if crs:
                    parks_polys = parks_polys.to_crs(crs)

            for geom in parks_polys.geometry:
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax.plot(x, y, zs=0, zdir='z', color=THEME['parks'], linewidth=0.5)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, zs=0, zdir='z', color=THEME['parks'], linewidth=0.5)

    # Layer 3: Roads (as lines at z=0)
    if G is not None and show_roads:
        print("  Drawing roads...")
        for u, v, data in G_proj.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
            else:
                xs = [G_proj.nodes[u]['x'], G_proj.nodes[v]['x']]
                ys = [G_proj.nodes[u]['y'], G_proj.nodes[v]['y']]

            highway = data.get('highway', 'unclassified')
            if isinstance(highway, list):
                highway = highway[0] if highway else 'unclassified'

            # Get color based on road type
            if highway in ['motorway', 'motorway_link']:
                color = THEME['road_motorway']
                width_val = 1.0
            elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
                color = THEME['road_primary']
                width_val = 0.8
            elif highway in ['secondary', 'secondary_link']:
                color = THEME['road_secondary']
                width_val = 0.6
            elif highway in ['tertiary', 'tertiary_link']:
                color = THEME['road_tertiary']
                width_val = 0.4
            else:
                color = THEME['road_residential']
                width_val = 0.3

            ax.plot(xs, ys, zs=0, zdir='z', color=color, linewidth=width_val)

    # Layer 4: Buildings (3D extruded)
    if buildings is not None and not buildings.empty:
        print("  Creating building mesh...")
        try:
            buildings_proj = ox.projection.project_gdf(buildings)
        except Exception:
            if crs:
                buildings_proj = buildings.to_crs(crs)
            else:
                buildings_proj = buildings

        mesh = create_building_mesh(buildings_proj, THEME, max_buildings=max_buildings)
        if mesh is not None:
            ax.add_collection3d(mesh)

    # Calculate actual extents
    x_extent = xlim[1] - xlim[0]
    y_extent = ylim[1] - ylim[0]
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2

    # Apply zoom factor - shrink the visible area to zoom in
    zoomed_x_extent = x_extent / zoom
    zoomed_y_extent = y_extent / zoom

    # Set zoomed axis limits
    ax.set_xlim(x_center - zoomed_x_extent/2, x_center + zoomed_x_extent/2)
    ax.set_ylim(y_center - zoomed_y_extent/2, y_center + zoomed_y_extent/2)

    # Max building height for visualization
    max_z = 200
    ax.set_zlim(0, max_z)

    # Hide axes for cleaner look
    ax.set_axis_off()

    # Calculate proper Z aspect ratio based on zoomed extent
    z_aspect = max_z / zoomed_x_extent

    # Set aspect ratio with proper Z scaling
    ax.set_box_aspect([1, zoomed_y_extent/zoomed_x_extent, z_aspect])

    # Adjust camera distance to fill frame better
    ax.dist = 8  # Lower value = closer/more zoomed (default is ~10)

    # Add gradient fades at top and bottom using figure-level axes
    # Bottom gradient
    gradient_ax_bottom = fig.add_axes([0, 0, 1, 0.25], zorder=10)
    gradient_ax_bottom.set_xlim(0, 1)
    gradient_ax_bottom.set_ylim(0, 1)
    gradient_ax_bottom.axis('off')
    gradient_vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient_img = np.hstack((gradient_vals, gradient_vals))
    rgb = mcolors.to_rgb(THEME['gradient_color'])
    gradient_colors = np.zeros((256, 4))
    gradient_colors[:, 0] = rgb[0]
    gradient_colors[:, 1] = rgb[1]
    gradient_colors[:, 2] = rgb[2]
    gradient_colors[:, 3] = np.linspace(1, 0, 256)
    gradient_cmap = mcolors.ListedColormap(gradient_colors)
    gradient_ax_bottom.imshow(gradient_img, extent=[0, 1, 0, 1], aspect='auto',
                               cmap=gradient_cmap, origin='lower')

    # Top gradient
    gradient_ax_top = fig.add_axes([0, 0.85, 1, 0.15], zorder=10)
    gradient_ax_top.set_xlim(0, 1)
    gradient_ax_top.set_ylim(0, 1)
    gradient_ax_top.axis('off')
    gradient_colors_top = np.zeros((256, 4))
    gradient_colors_top[:, 0] = rgb[0]
    gradient_colors_top[:, 1] = rgb[1]
    gradient_colors_top[:, 2] = rgb[2]
    gradient_colors_top[:, 3] = np.linspace(0, 1, 256)
    gradient_cmap_top = mcolors.ListedColormap(gradient_colors_top)
    gradient_ax_top.imshow(gradient_img, extent=[0, 1, 0, 1], aspect='auto',
                            cmap=gradient_cmap_top, origin='lower')

    # Typography - match 2D version style
    scale_factor = width / 12.0
    BASE_MAIN = 60
    BASE_SUB = 22
    BASE_COORDS = 14
    BASE_ATTR = 8

    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=BASE_MAIN * scale_factor)
        font_sub = FontProperties(fname=FONTS['light'], size=BASE_SUB * scale_factor)
        font_coords = FontProperties(fname=FONTS['regular'], size=BASE_COORDS * scale_factor)
        font_attr = FontProperties(fname=FONTS['light'], size=BASE_ATTR * scale_factor)
    else:
        font_main = FontProperties(family='monospace', weight='bold', size=BASE_MAIN * scale_factor)
        font_sub = FontProperties(family='monospace', weight='normal', size=BASE_SUB * scale_factor)
        font_coords = FontProperties(family='monospace', size=BASE_COORDS * scale_factor)
        font_attr = FontProperties(family='monospace', size=BASE_ATTR * scale_factor)

    # Spaced city name like 2D version
    spaced_city = "  ".join(list(city.upper()))

    # Adjust font size for long city names
    base_adjusted_main = BASE_MAIN * scale_factor
    city_char_count = len(city)
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if FONTS:
        font_main_adjusted = FontProperties(fname=FONTS['bold'], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family='monospace', weight='bold', size=adjusted_font_size)

    # Text overlay (using figure coordinates)
    fig.text(0.5, 0.14, spaced_city, ha='center', va='bottom',
             fontproperties=font_main_adjusted, color=THEME['text'], zorder=11)

    # Decorative line
    line_ax = fig.add_axes([0.4, 0.125, 0.2, 0.001], zorder=11)
    line_ax.axhline(y=0.5, color=THEME['text'], linewidth=1 * scale_factor)
    line_ax.axis('off')

    country_text = country_label if country_label else country
    fig.text(0.5, 0.10, country_text.upper(), ha='center', va='bottom',
             fontproperties=font_sub, color=THEME['text'], zorder=11)

    # Coordinates
    lat, lon = point
    coords_text = f"{lat:.4f} N / {lon:.4f} E" if lat >= 0 else f"{abs(lat):.4f} S / {lon:.4f} E"
    if lon < 0:
        coords_text = coords_text.replace("E", "W")
    fig.text(0.5, 0.07, coords_text, ha='center', va='bottom',
             fontproperties=font_coords, color=THEME['text'], alpha=0.7, zorder=11)

    # Attribution
    fig.text(0.98, 0.02, "OpenStreetMap contributors", ha='right', va='bottom',
             fontproperties=font_attr, color=THEME['text'], alpha=0.5, zorder=11)

    # Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05)

    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)
    plt.close()

    print(f"[OK] Done! 3D poster saved as {output_file}")


def create_poster(city, country, point, dist, output_file, output_format, width=12, height=16, country_label=None, name_label=None):
    print(f"\nGenerating map for {city}, {country}...")
    
    # Progress bar for data fetching
    with tqdm(total=3, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        compensated_dist = dist * (max(height, width) / min(height, width))/4 # To compensate for viewport crop
        G = fetch_graph(point, compensated_dist)
        if G is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)
        
        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        water = fetch_features(point, compensated_dist, tags={'natural': 'water', 'waterway': 'riverbank'}, name='water')
        pbar.update(1)
        
        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(point, compensated_dist, tags={'leisure': 'park', 'landuse': 'grass'}, name='parks')
        pbar.update(1)
    
    print("[OK] All data retrieved successfully!")
    
    # 2. Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    # Project graph to a metric CRS so distances and aspect are linear (meters)
    G_proj = ox.project_graph(G)
    
    # 3. Plot Layers
    # Layer 1: Polygons (filter to only plot polygon/multipolygon geometries, not points)
    if water is not None and not water.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        water_polys = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polys.empty:
            # Project water features in the same CRS as the graph
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                water_polys = water_polys.to_crs(G_proj.graph['crs'])
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)
    
    if parks is not None and not parks.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        parks_polys = parks[parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not parks_polys.empty:
            # Project park features in the same CRS as the graph
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                parks_polys = parks_polys.to_crs(G_proj.graph['crs'])
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)
    
    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G_proj)
    edge_widths = get_edge_widths_by_type(G_proj)

    # Determine cropping limits to maintain the poster aspect ratio
    crop_xlim, crop_ylim = get_crop_limits(G_proj, point, fig, compensated_dist)
    # Plot the projected graph and then apply the cropped limits
    ox.plot_graph(
        G_proj, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False, close=False
    )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)
    
    # Layer 3: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)
    
    # Calculate scale factor based on poster width (reference width 12 inches)
    scale_factor = width / 12.0
    
    # Base font sizes (at 12 inches width)
    BASE_MAIN = 60
    BASE_TOP = 40
    BASE_SUB = 22
    BASE_COORDS = 14
    BASE_ATTR = 8
    
    # 4. Typography using Roboto font
    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=BASE_MAIN * scale_factor)
        font_top = FontProperties(fname=FONTS['bold'], size=BASE_TOP * scale_factor)
        font_sub = FontProperties(fname=FONTS['light'], size=BASE_SUB * scale_factor)
        font_coords = FontProperties(fname=FONTS['regular'], size=BASE_COORDS * scale_factor)
        font_attr = FontProperties(fname=FONTS['light'], size=BASE_ATTR * scale_factor)
    else:
        # Fallback to system fonts
        font_main = FontProperties(family='monospace', weight='bold', size=BASE_MAIN * scale_factor)
        font_top = FontProperties(family='monospace', weight='bold', size=BASE_TOP * scale_factor)
        font_sub = FontProperties(family='monospace', weight='normal', size=BASE_SUB * scale_factor)
        font_coords = FontProperties(family='monospace', size=BASE_COORDS * scale_factor)
        font_attr = FontProperties(family='monospace', size=BASE_ATTR * scale_factor)
    
    spaced_city = "  ".join(list(city.upper()))
    
    # Dynamically adjust font size based on city name length to prevent truncation
    # We use the already scaled "main" font size as the starting point.
    base_adjusted_main = BASE_MAIN * scale_factor
    city_char_count = len(city)
    
    # Heuristic: If length is > 10, start reducing.
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor) 
    else:
        adjusted_font_size = base_adjusted_main
    
    if FONTS:
        font_main_adjusted = FontProperties(fname=FONTS['bold'], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family='monospace', weight='bold', size=adjusted_font_size)

    # --- BOTTOM TEXT ---
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_main_adjusted, zorder=11)
    
    country_text = country_label if country_label is not None else country
    ax.text(0.5, 0.10, country_text.upper(), transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)
    
    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")
    
    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)
    
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
            color=THEME['text'], linewidth=1 * scale_factor, zorder=11)

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_attr = FontProperties(family='monospace', size=8)
    
    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=THEME['text'], alpha=0.5, ha='right', va='bottom', 
            fontproperties=font_attr, zorder=11)

    # 5. Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05,)

    # DPI matters mainly for raster formats
    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    print(f"[OK] Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  uv run create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  uv run create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  uv run create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid

  # Waterfront & canals
  uv run create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  uv run create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  uv run create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline

  # Radial patterns
  uv run create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  uv run create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads

  # Organic old cities
  uv run create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  uv run create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  uv run create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout

  # Coastal cities
  uv run create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  uv run create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  uv run create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula

  # River cities
  uv run create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  uv run create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split

  # 3D Building Visualization
  uv run create_map_poster.py -c "Manhattan" -C "USA" -t noir --3d -d 5000      # NYC skyline
  uv run create_map_poster.py -c "Paris" -C "France" --3d --elevation 45        # Higher camera angle
  uv run create_map_poster.py -c "Venice" -C "Italy" --3d --no-roads            # Buildings only
  uv run create_map_poster.py -c "Tokyo" -C "Japan" --3d --azimuth -45 -d 8000  # Different view angle

  # List themes
  uv run create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --country-label   Override country text displayed on poster
  --theme, -t       Theme name (default: feature_based)
  --all-themes      Generate posters for all themes
  --distance, -d    Map radius in meters (default: 29000)
  --list-themes     List all available themes

3D Mode Options:
  --3d, --buildings Enable 3D building visualization
  --elevation       Camera elevation angle in degrees (default: 30.0)
                    Low values (15-25) = dramatic street-level view
                    High values (50-70) = more top-down architectural view
  --azimuth         Camera rotation angle in degrees (default: -60.0)
  --zoom            Frame fill factor (default: 1.5, higher = more zoomed)
  --no-roads        Hide roads in 3D mode
  --no-water        Hide water features in 3D mode
  --no-parks        Hide parks/green spaces in 3D mode
  --max-buildings   Maximum buildings to render (default: 5000)

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run create_map_poster.py --city "New York" --country "USA"
  uv run create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  uv run create_map_poster.py --city Paris --country France --theme noir --distance 15000
  uv run create_map_poster.py --list-themes
        """
    )
    
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--country-label', dest='country_label', type=str, help='Override country text displayed on poster')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--all-themes', '--All-themes', dest='all_themes', action='store_true', help='Generate posters for all themes')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--width', '-W', type=float, default=12, help='Image width in inches (default: 12)')
    parser.add_argument('--height', '-H', type=float, default=16, help='Image height in inches (default: 16)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'svg', 'pdf'],help='Output format for the poster (default: png)')

    # 3D mode arguments
    parser.add_argument('--3d', '--buildings', dest='enable_3d', action='store_true',
                       help='Enable 3D building visualization mode')
    parser.add_argument('--elevation', type=float, default=30.0,
                       help='Camera elevation angle in degrees (default: 30.0)')
    parser.add_argument('--azimuth', type=float, default=-60.0,
                       help='Camera azimuth angle in degrees (default: -60.0)')
    parser.add_argument('--no-roads', dest='show_roads', action='store_false',
                       help='Hide roads in 3D mode')
    parser.add_argument('--no-water', dest='show_water', action='store_false',
                       help='Hide water features in 3D mode')
    parser.add_argument('--no-parks', dest='show_parks', action='store_false',
                       help='Hide parks/green spaces in 3D mode')
    parser.add_argument('--max-buildings', type=int, default=5000,
                       help='Maximum number of buildings to render (default: 5000)')
    parser.add_argument('--zoom', type=float, default=1.5,
                       help='Zoom factor to fill frame (default: 1.5, higher = more zoomed)')

    args = parser.parse_args()
    
    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)
    
    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)
    
    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)
    
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        os.sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            os.sys.exit(1)
        themes_to_generate = [args.theme]
    
    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)
    
    # Get coordinates and generate poster
    try:
        coords = get_coordinates(args.city, args.country)
        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)

            if args.enable_3d:
                # 3D mode with extruded buildings
                create_3d_poster(
                    args.city, args.country, coords, args.distance,
                    output_file, args.format, args.width, args.height,
                    country_label=args.country_label,
                    elevation=args.elevation,
                    azimuth=args.azimuth,
                    show_roads=args.show_roads,
                    show_water=args.show_water,
                    show_parks=args.show_parks,
                    max_buildings=args.max_buildings,
                    zoom=args.zoom
                )
            else:
                # Standard 2D mode
                create_poster(
                    args.city, args.country, coords, args.distance,
                    output_file, args.format, args.width, args.height,
                    country_label=args.country_label
                )

        print("\n" + "=" * 50)
        print("[OK] Poster generation complete!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nERROR: Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
