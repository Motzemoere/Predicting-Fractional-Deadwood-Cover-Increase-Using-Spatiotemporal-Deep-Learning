"""
Centralized Path Configuration for Deadwood Forecasting Pipeline
=================================================================

All paths are defined here. Import this module in any script that needs paths.

Usage:
    from config.paths import PATHS, DATA_SOURCES
    
    cubes_dir = PATHS.cubes
    era5_files = DATA_SOURCES["era5"]
"""

from pathlib import Path
from glob import glob
from dataclasses import dataclass
import os


# ============================================================================
# Base Paths
# ============================================================================

# Auto-detect project root (parent of config folder)
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent  # config/paths.py -> project root

# Main data 
DATA_DIR = PROJECT_ROOT / "data" 

# External data sources (shared/read-only)
EXTERNAL_DATA_ROOT = Path("/mnt/gsdata/projects")

# ============================================================================
# Structured Paths Object
# ============================================================================

@dataclass(frozen=True)
class ProjectPaths:
    """All project paths in one place."""
    
    # Project structure
    project_root: Path = PROJECT_ROOT
    
    # Large data directory (SSD - cubes, training sets, outputs)
    data_dir: Path = DATA_DIR
    cubes: Path = DATA_DIR / "cubes"
    training_sets: Path = DATA_DIR / "training_sets"
    training_runs: Path = DATA_DIR / "training_runs"
    logs: Path = DATA_DIR / "logs"
    predictions: Path = DATA_DIR / "predictions"
    
    # Config
    config_dir: Path = PROJECT_ROOT / "config"
    base_config: Path = PROJECT_ROOT / "config" / "base.yaml"
    experiments_dir: Path = PROJECT_ROOT / "config" / "experiments"
    
    # Metadata (small files, version controlled in project)
    meta_data_dir: Path = DATA_DIR / "meta_data"
    
    # Figures
    figs_dir: Path = DATA_DIR / "figs"
    
    def __post_init__(self):
        """Ensure critical directories exist."""
        for path in [self.cubes, self.training_sets, self.training_runs, self.logs]:
            path.mkdir(parents=True, exist_ok=True)
# Singleton instance
PATHS = ProjectPaths()


# ============================================================================
# External Data Sources (Raw/Read-Only)
# ============================================================================

DATA_SOURCES = {
    "worldclim": sorted([
        Path(p) for p in glob(str(
            EXTERNAL_DATA_ROOT / "panops/panops-data-registry/data/worldclim/wc2-1_30s_bio/*.tif"
        ))
    ]),
    
    "soilgrids": sorted([
        Path(p) for p in glob(str(
            EXTERNAL_DATA_ROOT / "panops/panops-data-registry/data/soilgrids/soilgrids_v2-0_1km/*.tif"
        ))
    ]),
    
    "era5": [
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/era5/germany/weekly/t2m/t2m_weekly.zarr",
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/era5/germany/weekly/tp/tp_weekly.zarr",
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/era5/germany/weekly/d2m/d2m_weekly.zarr",
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/era5/germany/weekly/ssrd/ssrd_weekly.zarr",
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/era5/germany/weekly/lai_hv/lai_hv_weekly.zarr",
    ],
    
    "dem_aspect_slope": [
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/copernicus30dem/dem_derivatives/slope.tif",
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/copernicus30dem/dem_derivatives/aspect.tif",
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/copernicus30dem/cop_dem_data/dem.tif",
    ],
    
    "canopy_height": [
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/ETH_GlobalCanopyHeight_10m_2020_v1/609802/ETH_GlobalCanopyHeight_10m_2020_mosaic_Map.vrt",
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/ETH_GlobalCanopyHeight_10m_2020_v1/609802/ETH_GlobalCanopyHeight_10m_2020_mosaic_Map_SD.vrt",
    ],
    
    "forest_type": [
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/PROBAV_Forest_Type_2019.tif"
    ],
    
    "stand_age": [
        EXTERNAL_DATA_ROOT / "other/deadtrees_forecasting_data/BGIForestAgeMPIBGC1.0.0.nc"
    ]
}

# ============================================================================
# S3 Remote Configuration
# ============================================================================

S3_CONFIG = {
    "run_id": "run_v1004_v1000_crop_half_fold_None_checkpoint_199",
    "pred_dir": "frct-sentinel2/sentinel-2-inferences",
    "sentinel_dir": "frct-sentinel2/sentinel-2-cubes",
    "endpoint_url": "https://s3.bwsfs.uni-freiburg.de",
    "region_name": "fr2-ec82",
}


# ============================================================================
# Helper Functions
# ============================================================================

def validate_paths():
    """Check that critical paths exist. Call at startup for early failure."""
    issues = []
    
    if not PATHS.data_dir.exists():
        issues.append(f"Data directory not found: {PATHS.data_dir}")
    
    if not PATHS.meta_data_dir.exists():
        issues.append(f"Metadata directory not found: {PATHS.meta_data_dir}")
    
    # Check external data sources
    for source_name, paths in DATA_SOURCES.items():
        if not paths:
            issues.append(f"No files found for {source_name}")
        elif not paths[0].exists():
            issues.append(f"First {source_name} file not found: {paths[0]}")
    
    if issues:
        print("⚠️  Path validation warnings:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✓ All paths validated")
    
    return len(issues) == 0


# ============================================================================
# Quick test when run directly
# ============================================================================

if __name__ == "__main__":
    print(f"Project root: {PATHS.project_root}")
    print(f"Data directory: {PATHS.data_dir}")
    print(f"Cubes directory: {PATHS.cubes}")
    print(f"\nExternal data sources:")
    for name, paths in DATA_SOURCES.items():
        print(f"  {name}: {len(paths)} files")
    print()
    validate_paths()
