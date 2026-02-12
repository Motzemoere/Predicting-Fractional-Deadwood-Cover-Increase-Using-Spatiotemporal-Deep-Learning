import geopandas as gpd
import folium
import sys
from pathlib import Path
from shapely.geometry import box
import pandas as pd
from shapely.geometry import Point
import ast
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

def to_utm_coords(lon, lat, from_crs='EPSG:4326', utm_zone=32, northern=True):
    """
    Convert (lon, lat) from any CRS (default WGS84) to UTM coordinates.
    Returns (utm_x, utm_y, utm_epsg_code)
    """
    import pyproj
    # Determine UTM EPSG code
    if northern:
        utm_epsg = 32600 + utm_zone
    else:
        utm_epsg = 32700 + utm_zone
    proj_from = pyproj.CRS(from_crs)
    proj_to = pyproj.CRS(f'EPSG:{utm_epsg}')
    transformer = pyproj.Transformer.from_crs(proj_from, proj_to, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_epsg

def create_aoi(minx, miny, maxx, maxy, crs_from, crs_to):
    poly = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame(index=[0], crs=crs_from, geometry=[poly])
    return gdf.to_crs(crs_to)

def add_aoi_to_map(m, geometry, color, popup):
    folium.GeoJson(
        geometry.__geo_interface__,
        style_function=lambda x: {
            'fillColor': color,
            'color': color,
            'weight': 3,
            'fillOpacity': 0.2
        },
        popup=popup
    ).add_to(m)

def plot_cube_on_map(gdf_meta, target_cube_id):
    """
    Plots the specific cube geometry and its original point of interest on a Folium map.
    
    Args:
        gdf_meta (gpd.GeoDataFrame): The metadata dataframe containing the cubes.
        target_cube_id (int): The ID of the cube to visualize.
        
    Returns:
        folium.Map: The map object.
    """
    # 1. Select the specific row
    subset = gdf_meta[gdf_meta['cube_id'] == target_cube_id]
    
    if subset.empty:
        print(f"Error: Cube ID {target_cube_id} not found.")
        return None
    
    row = subset.iloc[0]
    
    # 2. Process the Cube Geometry (Polygon)
    # The geometry is already in the GDF's CRS (EPSG:25832), convert to WGS84 for Folium
    cube_geo_wgs84 = subset.geometry.to_crs("EPSG:4326").iloc[0]
    
    # 3. Process the Point of Interest (Point)
    # The 'point of interest' column might be loaded as a string " (x, y) " from file 
    # or a tuple object if still in memory.
    poi_raw = row['point of interest']
    
    if isinstance(poi_raw, str):
        # Safely convert string representation of tuple back to tuple
        poi_coords = ast.literal_eval(poi_raw)
    else:
        poi_coords = poi_raw
        
    # Extract the original EPSG from the grid_id (format: cube_{epsg}_{x}_{y})
    # This is necessary because the POI tuple coordinates are in the original Block CRS,
    # not necessarily the GDF Metadata CRS.
    try:
        original_epsg = int(row['grid_id'].split('_')[1])
        poi_crs = f"EPSG:{original_epsg}"
    except (IndexError, ValueError):
        # Fallback if grid_id format is unexpected
        print("Warning: Could not parse EPSG from grid_id. Assuming same as GDF.")
        poi_crs = gdf_meta.crs

    # Create a GeoSeries for the Point and convert to WGS84
    poi_point = Point(poi_coords[0], poi_coords[1])
    poi_gs = gpd.GeoSeries([poi_point], crs=poi_crs)
    poi_wgs84 = poi_gs.to_crs("EPSG:4326").iloc[0]

    # 4. Create the Map
    # Center map on the POI
    m = folium.Map(location=[poi_wgs84.y, poi_wgs84.x], zoom_start=13, tiles='OpenStreetMap')

    # Add the Cube Polygon (Red Square)
    folium.GeoJson(
        cube_geo_wgs84.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 2,
            'fillOpacity': 0.2
        },
        popup=f"Cube ID: {row['cube_id']}<br>Desc: {row['description']}"
    ).add_to(m)

    # Add the Point of Interest (Blue Marker)
    folium.Marker(
        location=[poi_wgs84.y, poi_wgs84.x],
        popup=f"Point of Interest<br>Original: {poi_coords}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    return m



if __name__ == "__main__":
    gdf = gpd.read_file(PATHS.meta_data_dir / "europe_inference_blocks.gpkg")
    locations = {
        'Feldsee': {'lon': 8.03254, 
                    'lat': 47.870452,
                    'description': 'Feldsee area in Black Forest, Germany'},
        'Pferdekopf': {'lon': 10.597962, 
                    'lat': 51.681113,
                    'description': 'Pferdekopf area in Harz National Park, Germany'},
        'Hoher_Draberg': {'lon': 8.395838, 
                        'lat': 48.693991,
                        'description': 'Hoher Draberg area in Black Forest, Germany'}
    }

    LOCATION = locations['Hoher_Draberg']
    # Get UTM coordinates
    utm_x, utm_y, utm_zone = to_utm_coords(LOCATION['lon'], LOCATION['lat'], from_crs='EPSG:4326', utm_zone=32, northern=True)
    gdf_zone32 = gdf[gdf['utm_epsg'] == utm_zone]

    blocks = gdf_zone32[
        (gdf_zone32['local_minx'] <= utm_x) & 
        (gdf_zone32['local_maxx'] >= utm_x) &
        (gdf_zone32['local_miny'] <= utm_y) & 
        (gdf_zone32['local_maxy'] >= utm_y)
    ]
    # Get the block (could be multiple)
    block = blocks.iloc[0]
    block_id = block['block_id']


    # Create map
    m = folium.Map(location=[LOCATION['lat'], LOCATION['lon']], zoom_start=13, tiles='OpenStreetMap')
    folium.GeoJson(
        block.geometry.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'green',
            'color': 'darkgreen',
            'weight': 2,
            'fillOpacity': 0.4
        },
        popup=f"Block: {block['block_id']}<br>Forest Cover: {block['forest_cover']:.2%}"
    ).add_to(m)
    folium.Marker(
        location=[LOCATION['lat'], LOCATION['lon']],
        popup=LOCATION['description'],
        icon=folium.Icon(color='red')
    ).add_to(m)

    m

    gdf_training_cubes = gpd.read_file(PATHS.data_dir / "training_sets/training_set_00_fd_<25/cube_set.gpkg")
    gdf_training_cubes_transformed = gdf_training_cubes.to_crs(gdf_zone32.crs)
    block_cubes = gdf_training_cubes_transformed[gdf_training_cubes_transformed.intersects(block.geometry)]
    
    for _, cube in block_cubes.iterrows():
        if cube['is_holdout']:
            color = 'green'
        else:
            color = 'orange'
        add_aoi_to_map(m, cube.geometry, color, f"Training Cube ID: {cube['cube_id']}")

    m

    # show cube which intersects the point
    for _, cube in gdf_training_cubes_transformed.iterrows():
        if cube.geometry.contains(Point(utm_x, utm_y)):
            add_aoi_to_map(m, cube.geometry, 'purple', f"Intersecting Cube ID: {cube['cube_id']}")


    gdf_holdout = gdf_training_cubes[gdf_training_cubes['is_holdout'] == True]
    len(gdf_holdout[gdf_holdout['mortality_class']=='high'])


    # Now we need to select a 5x5 km area of interest:
    side_length = 5000
    half_side = side_length / 2

    # Put in the top left corner of the area
    minx = utm_x - side_length
    maxx = utm_x 
    miny = utm_y - side_length
    maxy = utm_y 

    # Create AOI and adjusted AOI using helper
    aoi_cube = create_aoi(minx, miny, maxx, maxy, 'EPSG:32632', 'EPSG:4326')
    add_aoi_to_map(m, aoi_cube.geometry.iloc[0], 'blue', "5x5km AOI")
    m

    # Adjust 5x5 grid to preference
    minx_adj = minx + 500 + half_side 
    miny_adj = miny + 500
    maxx_adj = maxx + 500 + half_side
    maxy_adj = maxy + 500
    # For folium visualization (WGS84)
    aoi_cube_adj_wgs84 = create_aoi(minx_adj, miny_adj, maxx_adj, maxy_adj, 'EPSG:32632', 'EPSG:4326')
    add_aoi_to_map(m, aoi_cube_adj_wgs84.geometry.iloc[0], 'red', "5x5km AOI Adjusted")
    m



    # Load Gdf if exists and add new row:
    META_CRS = "EPSG:25832"  # Standardized metadata CRS
    DATA_DIR = PATHS.data_dir / "specific_cubes"
    META_DIR = PATHS.meta_data_dir
    META_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = META_DIR / "specific_cubes_meta.gpkg"

    if meta_path.exists():
        gdf_meta = gpd.read_file(meta_path)
        gdf_meta['cube_id'] = pd.to_numeric(gdf_meta['cube_id'], errors='coerce')
        cube_id = int(gdf_meta['cube_id'].max() + 1) if not gdf_meta['cube_id'].isnull().all() else 1000
    else:
        # Initialize an empty GDF with the target Metadata CRS
        gdf_meta = gpd.GeoDataFrame(
            columns=['block_id', 'cube_id', 'description', 'grid_id', 'geometry'], 
            geometry='geometry', 
            crs=META_CRS
        )
        cube_id = 1000 # Start from 1000 for specific cubes


    # For metadata: geometry in block CRS (UTM)
    cube_geom_utm = box(minx_adj, miny_adj, maxx_adj, maxy_adj)
    minx_cube, miny_cube, maxx_cube, maxy_cube = cube_geom_utm.bounds
    epsg_code = block['utm_epsg']
    grid_id = f"cube_{epsg_code}_{int(minx_cube)}_{int(miny_cube)}"
    block_crs = f'EPSG:{epsg_code}'
    new_row_block_crs = gpd.GeoDataFrame([{
        'block_id': block_id,
        'cube_id': cube_id,
        'description': LOCATION['description'],
        'grid_id': grid_id,
        'point of interest': (utm_x, utm_y),
        'geometry': cube_geom_utm
    }], crs=block_crs)
    new_row_meta_crs = new_row_block_crs.to_crs(META_CRS)

    # check if overlaps with existing cubes
    target_polygon = new_row_meta_crs.geometry.iloc[0]

    gdf_all_training_cubes = gpd.read_file(PATHS.meta_data_dir / "balanced_cube_set.gpkg")
    overlap_mask = gdf_all_training_cubes.intersects(target_polygon)
    overlapping_cubes = gdf_all_training_cubes[overlap_mask]
    if not overlapping_cubes.empty:
        print("Warning: The new cube overlaps with existing training cubes:")
        print(overlapping_cubes[['cube_id', 'geometry']])

        fig, ax = plt.subplots(figsize=(10, 10))
        overlapping_cubes.plot(ax=ax, color='blue', alpha=0.3, edgecolor='black', label='Found Cubes')
        new_row_meta_crs.plot(ax=ax, color='red', alpha=0.3, edgecolor='red', label='Target Cube')
        for x, y, label in zip(overlapping_cubes.geometry.centroid.x, overlapping_cubes.geometry.centroid.y, overlapping_cubes.cube_id):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        plt.title(f"Overlap Analysis: {len(overlapping_cubes)} overlapping cubes found")
        plt.show()

    else:
        print("No overlap detected with existing training cubes.")


    gdf_meta = pd.concat([gdf_meta, new_row_meta_crs], ignore_index=True)
    gdf_meta.to_file(meta_path, driver="GPKG")
    print(f"Added metadata to {meta_path.name} for cube_id {cube_id}")



