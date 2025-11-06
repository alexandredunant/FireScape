import matplotlib
matplotlib.use('Agg')
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_spatial_extents(file1, file2, output_path):
    """
    Plots the spatial extents of two NetCDF files.
    """

    with xr.open_dataset(file1) as ds1, xr.open_dataset(file2) as ds2:
        # Extract bounding box from the first file
        lon1_min, lon1_max = ds1.x.min().item(), ds1.x.max().item()
        lat1_min, lat1_max = ds1.y.min().item(), ds1.y.max().item()

        # Extract bounding box from the second file
        lon2_min, lon2_max = ds2.x.min().item(), ds2.x.max().item()
        lat2_min, lat2_max = ds2.y.min().item(), ds2.y.max().item()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot extent of file 1
        ax.add_patch(
            plt.Rectangle(
                (lon1_min, lat1_min),
                lon1_max - lon1_min,
                lat1_max - lat1_min,
                fill=True,
                color="blue",
                alpha=0.5,
                label="File 1 Extent",
            )
        )

        # Plot extent of file 2
        ax.add_patch(
            plt.Rectangle(
                (lon2_min, lat2_min),
                lon2_max - lon2_min,
                lat2_max - lat2_min,
                fill=True,
                color="red",
                alpha=0.5,
                label="File 2 Extent",
            )
        )

        ax.set_xlim(min(lon1_min, lon2_min) - 10000, max(lon1_max, lon2_max) + 10000)
        ax.set_ylim(min(lat1_min, lat2_min) - 10000, max(lat1_max, lat2_max) + 10000)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Spatial Extent Comparison")
        ax.legend()
        ax.grid(True)

        plt.savefig(output_path)
        plt.close()


def plot_time_series_comparison(file1, file2, output_path):
    """
    Plots the correlation of mean, min, and max time series for two NetCDF files.
    """
    with (
        xr.open_dataset(file1, chunks="auto") as ds1,
        xr.open_dataset(file2, chunks="auto") as ds2,
    ):
        # Calculate aggregates for file 1
        mean1 = ds1["pr"].mean(dim=["x", "y"]).compute()
        min1 = ds1["pr"].min(dim=["x", "y"]).compute()
        max1 = ds1["pr"].max(dim=["x", "y"]).compute()

        # Calculate aggregates for file 2
        mean2 = ds2["pr"].mean(dim=["x", "y"]).compute()
        min2 = ds2["pr"].min(dim=["x", "y"]).compute()
        max2 = ds2["pr"].max(dim=["x", "y"]).compute()

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # Data for plotting
        plot_data = [
            {"title": "Mean", "data1": mean1, "data2": mean2, "color": "blue"},
            {"title": "Min", "data1": min1, "data2": min2, "color": "green"},
            {"title": "Max", "data1": max1, "data2": max2, "color": "red"},
        ]

        for ax, p_data in zip(axes, plot_data):
            ax.scatter(
                p_data["data1"],
                p_data["data2"],
                label=p_data["title"],
                color=p_data["color"],
                alpha=0.5,
            )

            # Add 1:1 line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)

            ax.set_xlabel(f"File 1 {p_data['title']} Precipitation")
            ax.set_ylabel(f"File 2 {p_data['title']} Precipitation")
            ax.set_title(f"{p_data['title']} Precipitation Correlation")
            ax.legend()
            ax.grid(True)
            ax.set_aspect("equal", "box")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    file1_path = "/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp45/pr_EUR-11_pctl50_rcp45.nc"
    file2_path = "/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/climate_projections_ensemble_quantiles/precipitation/rcp45/pr_EUR-11_pctl50_rcp45.nc"
    output_dir = Path("/mnt/CEPH_PROJECTS/Firescape/01_Data_Preparation/")

    # Plot spatial extents
    spatial_plot_path = output_dir / "spatial_extent_comparison.png"
    plot_spatial_extents(file1_path, file2_path, spatial_plot_path)
    print(f"Spatial extent plot saved to: {spatial_plot_path}")

    # Plot time series comparison
    timeseries_plot_path = output_dir / "timeseries_comparison.png"
    plot_time_series_comparison(file1_path, file2_path, timeseries_plot_path)
    print(f"Time series comparison plot saved to: {timeseries_plot_path}")
