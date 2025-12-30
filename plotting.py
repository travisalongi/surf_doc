#!/usr/bin/env python
"""
This module contains functions to aid in plotting the data.

Author: Travis Alongi (talongi@usgs.gov)
"""

import numpy as np
import pandas as pd
import pyvista as pv
import rasterio
import matplotlib.colors as mplcolors
from pathlib import Path
from surf.utils import convert_latlon_to_utm
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling


def cb_friendly_sequential_cool_colormap(n_segments):
    """
    Generates a color-blind friendly sequential color map
    For plotting using matploblib

    Args:
        n_segments (int): Number of subdivisions of colormap.

    Returns:
        array: Array of RGB colors with shape (n_segments, 4).
            
    Note:
        Created using https://colorbrewwer2.org
    """
    colors = ["#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"]
    cmap = mplcolors.LinearSegmentedColormap.from_list("", colors)
    custom_colors = cmap(np.linspace(0, 1, n_segments))
    return custom_colors


def get_xy_coords_raster(geotiff_file):
    """
    Extracts x, y coordinates grids from a GeoTIFF file.

    Args:
        geotiff_file (str) : Path to GeoTIFF file.
    Returns:
        tuple: Two 2D numpy arays representing x and y coordinates.
    """
    with rasterio.open(geotiff_file) as src:
        # Read the raster data into a numpy array
        data = src.read(1)  # Read the first band

        # Get the affine transform of the raster (it contains geographic info)
        transform = src.transform

    # Get the size of the raster
    nrows, ncols = data.shape

    # Create the x, y coordinates based on the affine transform
    x = np.linspace(transform[2], transform[2] + transform[0] * ncols, ncols)
    y = np.linspace(transform[5], transform[5] + transform[4] * nrows, nrows)

    x, y = np.meshgrid(x, y)

    return x, y


def plot_base_map(plotter, base_map_file,opacity=0.2):
    """
    Plots a base map using a GeoTIFF file in PyVista.

    Args:
        plotter (pv.Plotter): PyVista plotter instance.
        base_map_file (str): Path to the base map GeoTIFF file.

    Example:
        ```python
        import pyvista as pv
        plotter=pv.Plotter()
        plot_base_map(plotter, 'base_map.tiff')
        ```
    """
    gtiff = pv.read(base_map_file)
    x, y = get_xy_coords_raster(base_map_file)

    if y.mean() < 90:
        x, y = convert_latlon_to_utm(x, y)

    # return x, y

    # Prepare the data for plotting
    mesh = np.column_stack([x.flatten(), y.flatten(), np.zeros_like(x).flatten()])
    pdata = pv.PolyData(mesh)

    # Extract transparency values and use that for opacity
    gts = gtiff["Tiff Scalars"]
    pdata["orig_sphere"] = gts[:, 0]
    trans = gts[:,3]
    print(np.sum(trans==0)/len(trans))
    if np.all(trans == 0):  # This means there are transparent pixels
        print("Map has NO transparent pixels")
        plotter.add_mesh(pdata, cmap="binary_r", opacity=opacity)
    else:
        print("Map has some transparent pixels")
        trans = trans/trans.max()
        trans = np.where(trans == 0, 1, 0)  # Swap 0s and 1s
        plotter.add_mesh(pdata, cmap="binary_r", opacity=trans)


def plot_fault_surfaces(plotter, ax, files,colors=None):
    """
    Plots 3D fault surfaces and their 2D projections.

    Args:
        plotter (pv.Plotter): PyVista plotter instance.
        ax (matplotlib.axes.Axes): Matplotlib axis for 2D projection.
        files (list): List of file paths to fault surface data (CSV format).
    """

    if colors is None:
        colors = cb_friendly_sequential_cool_colormap(len(files))

    for i, file in enumerate(files):
        df = pd.read_csv(file)
        coords = df[["x", "y", "z"]].to_numpy()

        # Plot 2D projection
        ax.plot(
            df.x,
            df.y,
            ".",
            color=colors[i],
            alpha=0.3,
            label=Path(file).stem,
        )
        ax.text(
            df.iloc[0].x,
            df.iloc[0].y,
            Path(file).stem,
            fontsize=14,
            rotation=30,
            bbox=dict(
                boxstyle="round", facecolor="whitesmoke", edgecolor=colors[i], alpha=0.8
            ),
        )
        ax.set_ylabel('UTM northing [m]')
        ax.set_xlabel('UTM easting [m]')
        ax.set_title('Fault Surf ID Map')
        ax.axis('equal')

        # Generate and plot 3D surface
        cloud = pv.PolyData(coords)
        surf = cloud.delaunay_2d()
        plotter.add_mesh(surf, color=colors[i], opacity=0.9)


def plot_earthquakes(plotter, eq_catalog, point_size=3, min_depth=-15000, color="pink"):
    """
    Plots earthquake locations as spheres in PyVista.

    Args:
        plotter (pv.Plotter): PyVista plotter instance.
        eq_catalog (str): Path to the earthquake catalog CSV file.
        point_size (int, optional): Size of plotted earthquake points. Default is 3.
        min_depth (float, optional): Minimum depth filter (meters). Default is -15000.
        color (str, optional): Color of plotted earthquakes. Default is "pink".

    Example:
        ```
        import pandas as pd
        import numpy as np
        catalog = pd.read_csv('catalog.csv')
        data = np.array([catalog.x, catalog.y, catalog.depth_m]).T
        plot_earthquakes(plotter, data, point_size=5, min_depth=-25000)
        ```
    """
    eq_df = pd.read_csv(eq_catalog)
    eq_df = eq_df[eq_df.depth_m > min_depth]
    eq_data = eq_df[["x", "y", "depth_m"]].to_numpy()

    plotter.add_mesh(
        eq_data,
        render_points_as_spheres=True,
        color=color,
        point_size=point_size,
        opacity=0.7,
    )


def plot_topo(
    plotter,
    geotiff_files,
    vert_exag=2,
    downsample=5,
    opacity=0.90,
    monochromatic=False,
    min_elev=-10000,
    max_elev=15000,
    clims=[-10000,10000]
):
    """
    Plots topography from multiple GeoTIFF tiles in a PyVista plotter.

    Merges elevation data from GeoTIFF files, converts coordinates to UTM, and 
    renders a 3D topographic surface with optional vertical exaggeration and coloring.

    Args:
        plotter (pv.Plotter): PyVista plotter instance for rendering.
        geotiff_files (list of str): Paths to GeoTIFF tiles containing elevation data.
        vert_exag (float, optional): Vertical exaggeration factor (default: 2).
        downsample (int, optional): Factor to reduce resolution for performance (default: 5).
        opacity (float, optional): Surface opacity (0 to 1, default: 0.90).
        monochromatic (bool, optional): If True, plots in grayscale; otherwise, uses a terrain colormap (default: False).
        min_elev (float, optional): Minimum elevation threshold (default: -10,000).
        max_elev (float, optional): Maximum elevation threshold (default: 15,000).

    Returns:
        tuple: (utm_x, utm_y, elevation, grid)
            - utm_x (numpy.ndarray): UTM x-coordinates of the mesh.
            - utm_y (numpy.ndarray): UTM y-coordinates of the mesh.
            - elevation (numpy.ndarray): Processed elevation data.
            - grid (pv.StructuredGrid): PyVista grid object of the topography.

    Note:
        - Assumes GeoTIFFs are in EPSG:4326 (WGS84 latitude/longitude).
        - Automatically determines the appropriate UTM zone.
        - Uses `pyproj` for coordinate transformation and `rasterio` for reading GeoTIFFs.

    Example:
        ```python
        import pyvista as pv
        plotter = pv.Plotter()
        geotiff_files = ["tile1.tif", "tile2.tif"]
        plot_topo(plotter, geotiff_files, vert_exag=2, monochromatic=False)
        plotter.show()
        ```
    """
    # Open all datasets
    datasets = [rasterio.open(f) for f in geotiff_files]

    # Merge all tiles into a single array
    merged_array, merged_transform = merge(datasets)

    # Get merged dataset properties
    crs = datasets[0].crs  # Assume all tiles have the same CRS
    width, height = merged_array.shape[2], merged_array.shape[1]  # Raster shape
    elevation = merged_array[0]  # First band (assumed to be elevation)
    elevation = np.where(elevation > min_elev, elevation, np.nan)
    elevation = np.where(elevation < max_elev, elevation, np.nan)

    # Close datasets
    for ds in datasets:
        ds.close()

    # Generate X, Y coordinates
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Downsample
    x = x[::downsample, ::downsample]
    y = y[::downsample, ::downsample]
    elevation = elevation[::downsample, ::downsample]

    lon, lat = merged_transform * (x, y)

    utm_x, utm_y = convert_latlon_to_utm(lat, lon)

    grid = pv.StructuredGrid(utm_x, utm_y, elevation * vert_exag)
    grid["Elevation"] = (elevation.T).ravel()
    if monochromatic is True:
        plotter.add_mesh(grid, color="yellow", show_edges=False, opacity=opacity)
    else:
        plotter.add_mesh(
            grid,
            scalars=grid["Elevation"],
            # cmap="terrain",
            cmap="gist_earth",
            show_edges=False,
            opacity=opacity,
            # clim=[-np.nanmax(elevation), np.nanmax(elevation)],
            clim=clims,
        )

    return utm_x, utm_y, elevation, grid


def plot_topo_with_texture(
    plotter,
    dem_files,
    texture_file,
    vert_exag=3,
    downsample=5,
    opacity=0.9,
    no_data_color=(128, 128, 128),
    min_elev=-10000,
    max_elev=20000,
):
    """
    Plots topography with an overlaid GeoTIFF texture.

    Args:
        plotter (pv.Plotter): PyVista plotter instance.
        dem_files (list of str): List of DEM GeoTIFF file paths.
        texture_file (str): Path to the GeoTIFF texture file.
        vert_exag (float, optional): Vertical exaggeration factor. Default is 1.5.
        downsample (int, optional): Factor to downsample the topography. Default is 3.
        opacity (float, optional): Opacity of the plotted mesh. Default is 0.9.
        no_data_color (tuple, optional): RGB color for no-data areas. Default is (128, 128, 128).
        min_elev (float, optional): Minimum elevation cutoff. Default is -10000.
        max_elev (float, optional): Maximum elevation cutoff. Default is 20000.

    Returns:
        pv.Texture:
            -PyVista texture object.

    Example:
        ```python
        plotter = pv.Plotter()
        dem_files = ['dem_tile1.tif', 'dem_tile2.tif']
        texture_file = 'texture.tif'
        plot_topo_with_texture(plotter, dem_files, texture_file)
        ```
    """
    # Merge DEM tiles
    dem_datasets = [rasterio.open(f) for f in dem_files]
    dem_array, dem_transform = merge(dem_datasets)
    elevation = dem_array[0]  # First band (assumed to be elevation)
    elevation = np.where(elevation > min_elev, elevation, 0)
    elevation = np.where(elevation < max_elev, elevation, 0)
    dem_crs = dem_datasets[0].crs

    # Read texture image
    with rasterio.open(texture_file) as src:
        img = src.read([1, 2, 3])  # Read RGB bands
        img_transform = src.transform
        img_crs = src.crs

    # Close DEM datasets
    for ds in dem_datasets:
        ds.close()

    # **Reproject texture to match DEM CRS**
    dst_transform = dem_transform  # Match DEM transform
    dst_shape = elevation.shape  # Match DEM shape
    reprojected_img = np.zeros((3, *dst_shape), dtype=np.uint8)  # Output buffer

    for i in range(3):  # Loop over RGB bands
        reproject(
            source=img[i],
            destination=reprojected_img[i],
            src_transform=img_transform,
            src_crs=img_crs,
            dst_transform=dst_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear,
        )

    # **Fix No-Data Areas (Replace Black Pixels with Gray)**
    mask = (
        (reprojected_img[0] == 0)
        & (reprojected_img[1] == 0)
        & (reprojected_img[2] == 0)
    )
    for i in range(3):  # Apply gray color
        reprojected_img[i][mask] = no_data_color[i]

    # Downsample elevation data
    height, width = elevation.shape
    elevation = elevation[::downsample, ::downsample]

    # Generate X, Y grid (pixel indices)
    x = np.arange(0, width, downsample)
    y = np.arange(0, height, downsample)
    x, y = np.meshgrid(x, y)

    # Convert to lat/lon
    lon, lat = dem_transform * (x, y)

    utm_x, utm_y = convert_latlon_to_utm(lat, lon)

    # Create PyVista StructuredGrid
    grid = pv.StructuredGrid(utm_x, utm_y, elevation * vert_exag)
    grid["Elevation"] = elevation.ravel()

    # **Generate texture coordinates using PyVista**
    grid.texture_map_to_plane(inplace=True)

    # Convert reprojected texture to PyVista format
    img_array = np.moveaxis(reprojected_img, 0, -1)  # Rearrange (H, W, C)
    img_array = img_array.astype(np.uint8)
    texture = pv.Texture(img_array)

    # Apply texture
    plotter.add_mesh(grid, texture=texture, opacity=opacity, show_edges=False)

    return texture
