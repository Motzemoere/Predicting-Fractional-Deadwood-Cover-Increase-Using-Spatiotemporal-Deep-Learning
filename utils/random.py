"""
Geospatial utilities for location enrichment.
Adds OpenStreetMap links and German administrative boundaries (Bundesländer) to cube metadata.
"""

import geopandas as gpd
import pandas as pd

def add_osm_links_and_bund(gdf, selected_locations=None):
    # Helper to choose the best available name column
    def _pick_name_column(gdf_obj, candidates):
        for c in candidates:
            if c in gdf_obj.columns:
                return c
        return None
    # 1) Centroids in WGS84 for linking and geocoding
    centroids = gdf.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids, crs=gdf.crs).to_crs(4326)

    # Build a proper GeoDataFrame with cube_id and geometry
    points_gdf = gpd.GeoDataFrame(
        {"cube_id": gdf["cube_id"].astype(int)},
        geometry=centroids_wgs84,
        crs=centroids_wgs84.crs,
    )

    # 2) Load Bundesländer polygons and match CRS
    bund_url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/3_mittel.geo.json"
    bund = gpd.read_file(bund_url)
    bund = bund.to_crs(points_gdf.crs)

    # 3) Spatial join: assign Bundesland to each centroid
    bund_name = _pick_name_column(bund, ["name", "GEN", "NAME_1", "NAME"])
    if bund_name is None:
        raise ValueError("Bundesland dataset missing a name column (expected one of: name, GEN, NAME_1, NAME)")
    joined = gpd.sjoin(
        points_gdf,
        bund[[bund_name, "geometry"]],
        how="left",
        predicate="intersects",
    ).rename(columns={bund_name: "federal_state"}).drop(columns=["index_right"])
    # 4) Build output table: cube_id, federal_state, lat, lon, OSM link
    out_df = pd.DataFrame({
        "cube_id": joined["cube_id"].astype(int),
        "federal_state": joined["federal_state"],
        "lat": joined.geometry.y,
        "lon": joined.geometry.x,
    })

    def format_latex_link(r):
        base_url = f"https://www.openstreetmap.org/?mlat={r.lat:.6f}&mlon={r.lon:.6f}"
        map_view = f"#map=15/{r.lat:.6f}/{r.lon:.6f}"
        full_url = (base_url + map_view).replace("#", "\\#")
        label = f"{int(r.cube_id)}\\_osm"
        return f"\\href{{{full_url}}}{{{label}}}"

    out_df["OSM_Link"] = out_df.apply(format_latex_link, axis=1)

    # Handle missing states
    out_df["federal_state"] = out_df["federal_state"].fillna("Unbekannt")

    # Optional ordering: selected cube IDs first (ascending), then the rest (ascending)
    if selected_locations:
        selected_set = {int(x) for x in selected_locations}
        out_df["__group"] = out_df["cube_id"].apply(lambda x: 0 if x in selected_set else 1)
        out_df = out_df.sort_values(["__group", "cube_id"]).drop(columns=["__group"]).reset_index(drop=True)
    else:
        out_df = out_df.sort_values("cube_id").reset_index(drop=True)

    return out_df
