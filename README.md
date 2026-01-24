# City Map Poster Generator

Generate beautiful, minimalist map posters for any city in the world.

<img src="posters/singapore_neon_cyberpunk_20260118_153328.png" width="250">
<img src="posters/dubai_midnight_blue_20260118_140807.png" width="250">

## Examples

|  Country  |     City      |     Theme      |                                    Poster                                    |
| :-------: | :-----------: | :------------: | :--------------------------------------------------------------------------: |
|    USA    | San Francisco |     sunset     |   <img src="posters/san_francisco_sunset_20260118_144726.png" width="250">   |
|   Spain   |   Barcelona   |   warm_beige   |   <img src="posters/barcelona_warm_beige_20260118_140048.png" width="250">   |
|   Italy   |    Venice     |   blueprint    |     <img src="posters/venice_blueprint_20260118_140505.png" width="250">     |
|   Japan   |     Tokyo     |  japanese_ink  |    <img src="posters/tokyo_japanese_ink_20260118_142446.png" width="250">    |
|   India   |    Mumbai     | contrast_zones |  <img src="posters/mumbai_contrast_zones_20260118_145843.png" width="250">   |
|  Morocco  |   Marrakech   |   terracotta   |   <img src="posters/marrakech_terracotta_20260118_143253.png" width="250">   |
| Singapore |   Singapore   | neon_cyberpunk | <img src="posters/singapore_neon_cyberpunk_20260118_153328.png" width="250"> |
| Australia |   Melbourne   |     forest     |     <img src="posters/melbourne_forest_20260118_153446.png" width="250">     |
|    UAE    |     Dubai     | midnight_blue  |   <img src="posters/dubai_midnight_blue_20260118_140807.png" width="250">    |

## Installation

```bash
uv sync
```

## Usage

```bash
uv run map.py --city <city> --country <country> [options]
```

### Options

| Option                          | Short | Description                                          | Default       |
| ------------------------------- | ----- | ---------------------------------------------------- | ------------- |
| `--city`                        | `-c`  | City name                                            | required      |
| `--country`                     | `-C`  | Country name                                         | required      |
| **OPTIONAL:** `--name`          |       | Override display name (city display on poster)       |               |
| **OPTIONAL:** `--country-label` |       | Override display country (country display on poster) |               |
| **OPTIONAL:** `--theme`         | `-t`  | Theme name                                           | feature_based |
| **OPTIONAL:** `--distance`      | `-d`  | Map radius in meters                                 | 29000         |
| **OPTIONAL:** `--list-themes`   |       | List all available themes                            |               |
| **OPTIONAL:** `--all-themes`    |       | Generate posters for all available themes            |               |
| **OPTIONAL:** `--width`         | `-W`  | Image width in inches                                | 12            |
| **OPTIONAL:** `--height`        | `-H`  | Image height in inches                               | 16            |

### 3D Building Mode Options

| Option                  | Description                              | Default  |
| ----------------------- | ---------------------------------------- | -------- |
| `--3d` or `--buildings` | Enable 3D building visualization         | false    |
| `--elevation`           | Camera elevation angle (15-70 degrees)   | 30.0     |
| `--azimuth`             | Camera rotation angle in degrees         | -60.0    |
| `--zoom`                | Frame fill factor (higher = more zoomed) | 1.5      |
| `--no-roads`            | Hide roads in 3D mode                    | false    |
| `--no-water`            | Hide water features                      | false    |
| `--no-parks`            | Hide parks/green spaces                  | false    |
| `--max-buildings`       | Limit buildings for performance          | no limit |

### Resolution Guide (300 DPI)

Use these values for `-W` and `-H` to target specific resolutions:

| Target               | Resolution (px) | Inches (-W / -H) |
| -------------------- | --------------- | ---------------- |
| **Instagram Post**   | 1080 x 1080     | 3.6 x 3.6        |
| **Mobile Wallpaper** | 1080 x 1920     | 3.6 x 6.4        |
| **HD Wallpaper**     | 1920 x 1080     | 6.4 x 3.6        |
| **4K Wallpaper**     | 3840 x 2160     | 12.8 x 7.2       |
| **A4 Print**         | 2480 x 3508     | 8.3 x 11.7       |

### Examples

```bash
# Iconic grid patterns
uv run map.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
uv run map.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district

# Waterfront & canals
uv run map.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
uv run map.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
uv run map.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline

# Radial patterns
uv run map.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
uv run map.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads

# Organic old cities
uv run map.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
uv run map.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
uv run map.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient layout

# Coastal cities
uv run map.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
uv run map.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
uv run map.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula

# River cities
uv run map.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
uv run map.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split

# 3D Building Visualization
uv run map.py -c "Manhattan" -C "USA" -t noir --3d -d 5000      # NYC skyline
uv run map.py -c "Paris" -C "France" --3d --elevation 45        # Higher camera angle
uv run map.py -c "Venice" -C "Italy" --3d --no-roads            # Buildings only
uv run map.py -c "Tokyo" -C "Japan" --3d --azimuth -45 -d 8000  # Different view angle
uv run map.py -c "Dubai" -C "UAE" --3d -t midnight_blue -d 8000 # Luxury skyline

# List available themes
uv run map.py --list-themes

# Generate posters for every theme
uv run map.py -c "Tokyo" -C "Japan" --all-themes
```

### Distance Guide

| Distance     | Best for                                           |
| ------------ | -------------------------------------------------- |
| 4000-6000m   | Small/dense cities (Venice, Amsterdam center)      |
| 8000-12000m  | Medium cities, focused downtown (Paris, Barcelona) |
| 15000-20000m | Large metros, full city view (Tokyo, Mumbai)       |

## 3D Building Visualization

The `--3d` flag enables true 3D building extrusion using OSM building data. Buildings are rendered as 3D boxes with heights derived from:

1. `height` tag (explicit height in meters)
2. `building:levels` tag (floors x 3.5m)
3. Building type inference (skyscrapers, offices, houses, etc.)
4. Default fallback (12m)

### Camera Controls

| Parameter        | Effect                            |
| ---------------- | --------------------------------- |
| `--elevation 30` | Default bird's-eye view           |
| `--elevation 60` | More top-down architectural view  |
| `--elevation 15` | Dramatic street-level perspective |
| `--azimuth -60`  | Default viewing direction         |
| `--azimuth 0`    | View from the south               |
| `--azimuth -90`  | View from the east                |
| `--zoom 1.5`     | Default frame fill                |
| `--zoom 2.0`     | Tighter crop, more zoomed in      |
| `--zoom 1.0`     | Wider view, more context          |

### Performance Tips

- Use smaller `--distance` values (3000-8000m) for 3D mode
- Reduce `--max-buildings` if rendering is slow
- Dense cities (Tokyo, NYC) may need lower building limits
- Use `--no-roads` for cleaner building-focused renders

## Themes

17 themes available in `themes/` directory:

| Theme             | Style                                     |
| ----------------- | ----------------------------------------- |
| `feature_based`   | Classic black & white with road hierarchy |
| `gradient_roads`  | Smooth gradient shading                   |
| `contrast_zones`  | High contrast urban density               |
| `noir`            | Pure black background, white roads        |
| `midnight_blue`   | Navy background with gold roads           |
| `blueprint`       | Architectural blueprint aesthetic         |
| `neon_cyberpunk`  | Dark with electric pink/cyan              |
| `warm_beige`      | Vintage sepia tones                       |
| `pastel_dream`    | Soft muted pastels                        |
| `japanese_ink`    | Minimalist ink wash style                 |
| `forest`          | Deep greens and sage                      |
| `ocean`           | Blues and teals for coastal cities        |
| `terracotta`      | Mediterranean warmth                      |
| `sunset`          | Warm oranges and pinks                    |
| `autumn`          | Seasonal burnt oranges and reds           |
| `copper_patina`   | Oxidized copper aesthetic                 |
| `monochrome_blue` | Single blue color family                  |

## Output

Posters are saved to `posters/` directory with format:

```
{city}_{theme}_{YYYYMMDD_HHMMSS}.png
```

## Adding Custom Themes

Create a JSON file in `themes/` directory:

```json
{
  "name": "My Theme",
  "description": "Description of the theme",
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
```

Building properties are optional (fallback defaults are provided) but recommended for 3D mode.

## Project Structure

```
map_poster/
├── map.py          # Main script
├── themes/               # Theme JSON files
├── fonts/                # Roboto font files
├── posters/              # Generated posters
└── README.md
```

## Hacker's Guide

Quick reference for contributors who want to extend or modify the script.

### Architecture Overview

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   CLI Parser    │────▶│  Geocoding   │────▶│  Data Fetching  │
│   (argparse)    │     │  (Nominatim) │     │    (OSMnx)      │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                     │
                        ┌──────────────┐             ▼
                        │    Output    │◀────┌─────────────────┐
                        │  (matplotlib)│     │   Rendering     │
                        └──────────────┘     │  (matplotlib)   │
                                             └─────────────────┘
```

### Key Functions

| Function                    | Purpose                        | Modify when...                |
| --------------------------- | ------------------------------ | ----------------------------- |
| `get_coordinates()`         | City to lat/lon via Nominatim  | Switching geocoding provider  |
| `create_poster()`           | Main 2D rendering pipeline     | Adding new map layers         |
| `create_3d_poster()`        | 3D building rendering pipeline | Modifying 3D visualization    |
| `fetch_buildings()`         | Fetch building data from OSM   | Changing building data source |
| `extract_building_height()` | Parse OSM height tags          | Adjusting height calculation  |
| `create_building_mesh()`    | Build 3D geometry              | Modifying building appearance |
| `get_edge_colors_by_type()` | Road color by OSM highway tag  | Changing road styling         |
| `get_edge_widths_by_type()` | Road width by importance       | Adjusting line weights        |
| `create_gradient_fade()`    | Top/bottom fade effect         | Modifying gradient overlay    |
| `load_theme()`              | JSON theme to dict             | Adding new theme properties   |

### Rendering Layers (z-order)

2D Mode:

```
z=11  Text labels (city, country, coords)
z=10  Gradient fades (top & bottom)
z=3   Roads (via ox.plot_graph)
z=2   Parks (green polygons)
z=1   Water (blue polygons)
z=0   Background color
```

3D Mode:

```
Layer 4: Buildings (Poly3DCollection, extruded)
Layer 3: Roads (3D lines at z=0)
Layer 2: Parks (3D lines at z=0)
Layer 1: Water (3D lines at z=0)
Text overlay via fig.text()
```

### OSM Highway Types → Road Hierarchy

```python
# In get_edge_colors_by_type() and get_edge_widths_by_type()
motorway, motorway_link     → Thickest (1.2), darkest
trunk, primary              → Thick (1.0)
secondary                   → Medium (0.8)
tertiary                    → Thin (0.6)
residential, living_street  → Thinnest (0.4), lightest
```

### Adding New Features

**New map layer (e.g., railways):**

```python
# In create_poster(), after parks fetch:
try:
    railways = ox.features_from_point(point, tags={'railway': 'rail'}, dist=dist)
except:
    railways = None

# Then plot before roads:
if railways is not None and not railways.empty:
    railways.plot(ax=ax, color=THEME['railway'], linewidth=0.5, zorder=2.5)
```

**New theme property:**

1. Add to theme JSON: `"railway": "#FF0000"`
2. Use in code: `THEME['railway']`
3. Add fallback in `load_theme()` default dict

### Typography Positioning

All text uses `transform=ax.transAxes` (0-1 normalized coordinates):

```
y=0.14  City name (spaced letters)
y=0.125 Decorative line
y=0.10  Country name
y=0.07  Coordinates
y=0.02  Attribution (bottom-right)
```

### Useful OSMnx Patterns

```python
# Get all buildings
buildings = ox.features_from_point(point, tags={'building': True}, dist=dist)

# Get specific amenities
cafes = ox.features_from_point(point, tags={'amenity': 'cafe'}, dist=dist)

# Different network types
G = ox.graph_from_point(point, dist=dist, network_type='drive')  # roads only
G = ox.graph_from_point(point, dist=dist, network_type='bike')   # bike paths
G = ox.graph_from_point(point, dist=dist, network_type='walk')   # pedestrian
```

### Performance Tips

- Large `dist` values (>20km) = slow downloads + memory heavy
- Cache coordinates locally to avoid Nominatim rate limits
- Use `network_type='drive'` instead of `'all'` for faster renders
- Reduce `dpi` from 300 to 150 for quick previews
