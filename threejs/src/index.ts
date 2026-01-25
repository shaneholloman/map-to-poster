import { chromium } from 'playwright';
import { program } from 'commander';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

interface Coordinates {
  lat: number;
  lon: number;
}

interface Building {
  coordinates: [number, number][];
  height: number;
  type?: string;
}

interface Theme {
  name: string;
  bg: string;
  text: string;
  building_fill: string;
  building_edge: string;
  road_primary: string;
  water: string;
  parks: string;
}

// Geocode city using Nominatim with cache fallback
async function geocode(city: string, country: string): Promise<Coordinates> {
  // Check for cached coordinates from Python version
  const cacheKey = `coords_${city.toLowerCase()}_${country.toLowerCase()}`;
  const cachePath = path.join(__dirname, '../../.cache', `${cacheKey}.pkl`);

  // Known coordinates fallback
  const knownCoords: Record<string, Coordinates> = {
    'wellington_new zealand': { lat: -41.288795, lon: 174.777211 },
    'manhattan_usa': { lat: 40.7831, lon: -73.9712 },
    'tokyo_japan': { lat: 35.6762, lon: 139.6503 },
    'paris_france': { lat: 48.8566, lon: 2.3522 },
    'london_uk': { lat: 51.5074, lon: -0.1278 },
  };

  const lookupKey = `${city.toLowerCase()}_${country.toLowerCase()}`;
  if (knownCoords[lookupKey]) {
    console.log(`  [CACHE] Using known coordinates`);
    return knownCoords[lookupKey];
  }

  const query = encodeURIComponent(`${city}, ${country}`);
  const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&limit=1`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(url, {
      headers: { 'User-Agent': 'map-poster-3d/1.0' },
      signal: controller.signal
    });

    clearTimeout(timeout);

    const data = await response.json() as Array<{ lat: string; lon: string }>;

    if (!data || data.length === 0) {
      throw new Error(`Could not geocode: ${city}, ${country}`);
    }

    return {
      lat: parseFloat(data[0].lat),
      lon: parseFloat(data[0].lon)
    };
  } catch (e) {
    console.log(`  [WARN] Nominatim failed, checking cache...`);

    if (knownCoords[lookupKey]) {
      return knownCoords[lookupKey];
    }

    throw new Error(`Could not geocode ${city}, ${country}: ${e}`);
  }
}

// Calculate bounding box from center point and radius
function getBbox(lat: number, lon: number, radiusMeters: number): string {
  const latPerMeter = 1 / 111320;
  const lonPerMeter = 1 / (111320 * Math.cos(lat * Math.PI / 180));

  const deltaLat = radiusMeters * latPerMeter;
  const deltaLon = radiusMeters * lonPerMeter;

  return `${lon - deltaLon},${lat - deltaLat},${lon + deltaLon},${lat + deltaLat}`;
}

// Fetch buildings from Overture Maps using CLI
async function fetchBuildings(lat: number, lon: number, radius: number): Promise<Building[]> {
  const bbox = getBbox(lat, lon, radius);
  const tempFile = path.join(__dirname, '../../.cache/temp_buildings.geojsonseq');

  // Ensure cache dir exists
  fs.mkdirSync(path.dirname(tempFile), { recursive: true });

  console.log(`[DOWNLOADING] Buildings from Overture Maps...`);
  console.log(`  Bbox: ${bbox}`);

  const { execSync } = await import('child_process');

  try {
    execSync(
      `uv run overturemaps download --bbox="${bbox}" -f geojsonseq -t building -o "${tempFile}"`,
      { stdio: 'inherit', cwd: path.join(__dirname, '../..') }
    );
  } catch (e) {
    console.error('Failed to download buildings:', e);
    return [];
  }

  // Parse GeoJSONSeq
  const buildings: Building[] = [];
  const content = fs.readFileSync(tempFile, 'utf-8');

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;

    try {
      const feature = JSON.parse(line);
      const geom = feature.geometry;
      const props = feature.properties || {};

      if (geom.type === 'Polygon') {
        const coords = geom.coordinates[0] as [number, number][];
        const height = props.height || (props.num_floors ? props.num_floors * 3.5 : 12);

        buildings.push({
          coordinates: coords,
          height: Math.min(height, 200),
          type: props.subtype || props.class
        });
      } else if (geom.type === 'MultiPolygon') {
        for (const polygon of geom.coordinates) {
          const coords = polygon[0] as [number, number][];
          const height = props.height || (props.num_floors ? props.num_floors * 3.5 : 12);

          buildings.push({
            coordinates: coords,
            height: Math.min(height, 200),
            type: props.subtype || props.class
          });
        }
      }
    } catch {
      continue;
    }
  }

  // Cleanup
  if (fs.existsSync(tempFile)) {
    fs.unlinkSync(tempFile);
  }

  console.log(`  [OK] Loaded ${buildings.length} buildings`);
  return buildings;
}

// Generate HTML with Three.js scene
function generateHtml(
  buildings: Building[],
  center: Coordinates,
  theme: Theme,
  options: {
    width: number;
    height: number;
    elevation: number;
    azimuth: number;
    city: string;
    country: string;
  }
): string {
  // Convert buildings to meters from center
  const buildingsData = buildings.map(b => {
    const coords = b.coordinates.map(([lon, lat]) => {
      const x = (lon - center.lon) * 111320 * Math.cos(center.lat * Math.PI / 180);
      const y = (lat - center.lat) * 111320;
      return [x, y];
    });
    return { coords, height: b.height };
  });

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { margin: 0; overflow: hidden; background: ${theme.bg}; }
    canvas { display: block; }
    #label {
      position: absolute;
      bottom: 10%;
      left: 0;
      right: 0;
      text-align: center;
      font-family: 'Roboto', 'Helvetica Neue', sans-serif;
      color: ${theme.text};
    }
    #city {
      font-size: 48px;
      font-weight: 900;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    #country {
      font-size: 18px;
      font-weight: 400;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      margin-top: 10px;
    }
  </style>
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;900&display=swap" rel="stylesheet">
</head>
<body>
  <div id="label">
    <div id="city">${options.city}</div>
    <div id="country">${options.country}</div>
  </div>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.170.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"
    }
  }
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const width = ${options.width};
    const height = ${options.height};
    const buildings = ${JSON.stringify(buildingsData)};

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('${theme.bg}');

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 50000);

    // Position camera based on elevation and azimuth
    const elevation = ${options.elevation} * Math.PI / 180;
    const azimuth = ${options.azimuth} * Math.PI / 180;
    const distance = 3000;

    camera.position.x = distance * Math.cos(elevation) * Math.sin(azimuth);
    camera.position.z = distance * Math.cos(elevation) * Math.cos(azimuth);
    camera.position.y = distance * Math.sin(elevation);
    camera.lookAt(0, 0, 0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(2); // Higher quality
    document.body.insertBefore(renderer.domElement, document.body.firstChild);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1000, 2000, 1000);
    scene.add(directionalLight);

    // Building material
    const buildingMaterial = new THREE.MeshLambertMaterial({
      color: '${theme.building_fill}',
      transparent: true,
      opacity: 0.9
    });

    const edgeMaterial = new THREE.LineBasicMaterial({
      color: '${theme.building_edge}',
      linewidth: 1
    });

    // Create buildings
    for (const building of buildings) {
      const shape = new THREE.Shape();
      const coords = building.coords;

      if (coords.length < 3) continue;

      shape.moveTo(coords[0][0], coords[0][1]);
      for (let i = 1; i < coords.length; i++) {
        shape.lineTo(coords[i][0], coords[i][1]);
      }
      shape.closePath();

      const extrudeSettings = {
        depth: building.height,
        bevelEnabled: false
      };

      const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
      geometry.rotateX(-Math.PI / 2);

      const mesh = new THREE.Mesh(geometry, buildingMaterial);
      scene.add(mesh);

      // Add edges
      const edges = new THREE.EdgesGeometry(geometry);
      const line = new THREE.LineSegments(edges, edgeMaterial);
      scene.add(line);
    }

    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(10000, 10000);
    const groundMaterial = new THREE.MeshBasicMaterial({
      color: '${theme.bg}',
      side: THREE.DoubleSide
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1;
    scene.add(ground);

    // Render
    renderer.render(scene, camera);

    // Signal ready for screenshot
    window.renderComplete = true;
  </script>
</body>
</html>`;
}

// Main function
async function main() {
  program
    .requiredOption('-c, --city <city>', 'City name')
    .requiredOption('-C, --country <country>', 'Country name')
    .option('-t, --theme <theme>', 'Theme name', 'noir')
    .option('-d, --distance <meters>', 'Map radius in meters', '1000')
    .option('--elevation <degrees>', 'Camera elevation angle', '30')
    .option('--azimuth <degrees>', 'Camera rotation angle', '-60')
    .option('-W, --width <pixels>', 'Image width', '3600')
    .option('-H, --height <pixels>', 'Image height', '4800')
    .option('-o, --output <file>', 'Output file path')
    .parse();

  const opts = program.opts();

  console.log('==================================================');
  console.log('3D Map Poster Generator (Three.js)');
  console.log('==================================================');

  // Load theme
  const themePath = path.join(__dirname, '../../themes', `${opts.theme}.json`);
  let theme: Theme;

  if (fs.existsSync(themePath)) {
    theme = JSON.parse(fs.readFileSync(themePath, 'utf-8'));
    console.log(`[OK] Loaded theme: ${theme.name}`);
  } else {
    // Default noir theme
    theme = {
      name: 'Noir',
      bg: '#0a0a0a',
      text: '#ffffff',
      building_fill: '#2a2a2a',
      building_edge: '#404040',
      road_primary: '#ffffff',
      water: '#1a1a2e',
      parks: '#1a2a1a'
    };
    console.log(`[WARN] Theme not found, using default noir`);
  }

  // Geocode
  console.log(`\nGeocoding ${opts.city}, ${opts.country}...`);
  const coords = await geocode(opts.city, opts.country);
  console.log(`  [OK] Coordinates: ${coords.lat.toFixed(6)}, ${coords.lon.toFixed(6)}`);

  // Fetch buildings
  const radius = parseInt(opts.distance);
  const buildings = await fetchBuildings(coords.lat, coords.lon, radius * 3);

  if (buildings.length === 0) {
    console.error('No buildings found. Exiting.');
    process.exit(1);
  }

  // Generate HTML
  console.log('\nGenerating Three.js scene...');
  const html = generateHtml(buildings, coords, theme, {
    width: parseInt(opts.width),
    height: parseInt(opts.height),
    elevation: parseFloat(opts.elevation),
    azimuth: parseFloat(opts.azimuth),
    city: opts.city,
    country: opts.country
  });

  const htmlPath = path.join(__dirname, '../../.cache/scene.html');
  fs.writeFileSync(htmlPath, html);
  console.log(`  [OK] Scene written to ${htmlPath}`);

  // Render with Playwright
  console.log('\nRendering with Playwright...');
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: {
      width: parseInt(opts.width),
      height: parseInt(opts.height)
    },
    deviceScaleFactor: 1
  });

  await page.goto(`file://${htmlPath}`);

  // Wait for render to complete
  await page.waitForFunction(() => (window as any).renderComplete === true, { timeout: 30000 });
  await page.waitForTimeout(500); // Extra time for fonts

  // Screenshot
  const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 15);
  const outputFile = opts.output ||
    path.join(__dirname, '../../posters', `${opts.city.toLowerCase().replace(/\s+/g, '_')}_${opts.theme}_3d_${timestamp}.png`);

  fs.mkdirSync(path.dirname(outputFile), { recursive: true });

  await page.screenshot({ path: outputFile, type: 'png' });
  await browser.close();

  const stats = fs.statSync(outputFile);
  console.log(`  [OK] Saved: ${outputFile} (${(stats.size / 1024 / 1024).toFixed(1)} MB)`);

  console.log('\n==================================================');
  console.log('[OK] Poster generation complete!');
  console.log('==================================================');
}

main().catch(console.error);
