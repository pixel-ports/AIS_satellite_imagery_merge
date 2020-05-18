import json
from osgeo import gdal, osr
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from dateutil import parser
import datetime
import numpy as np
from scipy.ndimage.morphology import binary_closing, binary_opening
from matplotlib.pyplot import imsave
import os
from glob import glob
import math
import shutil
from eolearn.core import LoadFromDisk


class DatasetFactory(object):

    def __init__(self, dataset_dir, out_folder="annotations", ann_file="annotations.json",
                 max_ship_length=30, cloud_thresh=0.2, ais_time_window=2, debug=True):
        """
        Constructor for DatasetFactory
        :param dataset_dir: Path to dataset base directory (e.g. PIXEL_EO)
        :param out_folder: Name of the folder created inside of each subdirectory (date) for annotations
        :param ann_file: Name of the file created inside 'out_folder' for annotations (json)
        :param max_ship_length: Annotate only ships longer than max_ship_length
        :param cloud_thresh: Annotations with cloud coverage < cloud_thresh are considered valid
        :param ais_time_window: Minutes +- around satellite imagery acquisition time
        :param debug: When True, image will be displayed with annotations and clouds
        """
        self.dataset_dir = dataset_dir
        self.out_folder = out_folder
        self.ann_file = ann_file
        self.ais_dirname = os.path.join(dataset_dir, "AIS_data/AIS_ASCII_by_UTM_Month")
        self.max_ship_length = max_ship_length
        self.cloud_thresh = cloud_thresh
        self.ais_time_window = ais_time_window
        self.debug = debug
        self.n_valid_planet = 0
        self.n_all_planet = 0

    def remove_annotations(self, start_date, end_date, providers=["planet_data"],
                           locations=["long_beach", "sf_bay"]):
        """
        This method will clear all the annotations (self.out_folder) in specified directories of interest
        :param start_date: Start date for annotations
        :param end_date: End date for annotations
        :param providers: List of providers (currently: planet_data and sentinel_data)
        :param locations: List of locations (currently: long_beach and sf_bay)
        :return: /
        """
        for provider in providers:
            for location in locations:
                for date in pd.date_range(start_date, end_date):
                    path_curr = os.path.join(self.dataset_dir, provider, location, "{:%Y_%m_%d}".format(date),
                                             self.out_folder)
                    if os.path.exists(path_curr):
                        print("Deleted: {}".format(path_curr))
                        shutil.rmtree(path_curr)

    def start_factory(self, start_date, end_date, providers=["planet_data"],
                      locations=["long_beach", "sf_bay"]):
        """
        This method will iterate through the dataset and will provide the data and call the methods,
        needed to annotate each particular folder(i.e. for a specific date, location and provider).
        AIS data is also loaded here in order to re-use it as much as possible.
        :param start_date: Start date for annotations
        :param end_date: End date for annotations
        :param providers: List of providers (currently: planet_data and sentinel_data)
        :param locations: List of locations (currently: long_beach and sf_bay)
        :return: /
        """

        for provider in providers:
            print("Started working on provider: " + provider)
            print(10 * "=")
            for location in locations:
                print("Started working on location: " + location)
                print(10 * "=")
                month_curr = start_date.month
                df = self.load_AIS(start_date, location, self.max_ship_length)
                for date in pd.date_range(start_date, end_date):
                    print("Processing: {:%Y-%m-%d}, {}, {} ".format(date, provider, location))

                    # new month -> load new AIS data
                    if date.month != month_curr:
                        month_curr = date.month
                        del df
                        df = self.load_AIS(date, location, self.max_ship_length)

                    path_curr = os.path.join(self.dataset_dir, provider, location, "{:%Y_%m_%d}".format(date))
                    if os.path.exists(path_curr):
                        if provider == "planet_data":
                            self.process_planet_folder(path_curr, df)
                        elif provider == "sentinel_data":
                            self.process_sentinel_folder(path_curr, df)
                        else:
                            print("Wrong provider specified! [{}]".format(provider))
                            exit(-1)
                    else:
                        continue
                        # print("Directory doesn't exists! [{}]".format(path_curr))

    def process_planet_folder(self, path, df_ais):
        """
        This method iterates through the content of each folder and stores all the annotations for this folder
        as gathered by a specific method (i.e. process_planet_image) to process each satellite imagery.
        :param path: Path to the folder of interest to annotate
        :param df_ais: AIS data for this month (filtered already by ship length and status)
        :return: /
        """
        annotations = []
        for json_filename in glob(os.path.join(path, "*.json")):
            image_id = os.path.splitext(os.path.basename(json_filename))[0]
            print(5 * "=")
            print("Started processing Planet imagery with ID: " + image_id)

            satellite_imagery = "{}_3B_Visual.tif".format(image_id)
            visual_filename = os.path.join(path, satellite_imagery)
            mask_filename = os.path.join(path, "{}_3B_Analytic_DN_udm.tif".format(image_id))

            print("\x1b[31mSatellite imagery: \x1b[0m" + visual_filename)
            print("\x1b[31mUDM file (cloud mask) \x1b[0m" + mask_filename)
            print("\x1b[31mJSON file (metadata) \x1b[0m" + json_filename)

            return_values = self.process_planet_image(visual_filename, mask_filename, json_filename, df_ais)

            if not return_values:
                print("\x1b[31mSkipping ID: {}!\x1b[0m".format(image_id))
                continue
            else:
                patches_real, ratios, fig, x, y, mmsi, lengths, widths, vessel_types, gdf_subset = return_values

            if len(ratios) == 0:
                continue

            annotation = {}
            out_path_patches_valid = os.path.join(path, self.out_folder, "patches_valid")
            out_path_patches_cloudy = os.path.join(path, self.out_folder, "patches_cloudy")

            if not os.path.exists(out_path_patches_valid):
                os.makedirs(out_path_patches_valid)

            if not os.path.exists(out_path_patches_cloudy):
                os.makedirs(out_path_patches_cloudy)

            for i in range(len(ratios)):

                if math.isnan(ratios[i]):
                    print("Invalid value of cloud cover - {}!".format(ratios[i]))
                    continue

                annotation[str(mmsi[i])] = {
                    "x": int(x[i]),
                    "y": int(y[i]),
                    "cloud_cover": ratios[i],
                    "length": lengths[mmsi[i]],
                    "width": widths[mmsi[i]],
                    "vessel_type": vessel_types[mmsi[i]]
                }
                patch = np.transpose(patches_real[i], [1, 2, 0])
                patch_path = os.path.join(out_path_patches_valid if ratios[i] < self.cloud_thresh else
                                          out_path_patches_cloudy, "{}_{}_{}.png".format(image_id, mmsi[i],
                                                                                         int(ratios[i] * 100)))
                imsave(patch_path, patch)

            annotations.append(annotation)
            with open(os.path.join(path, self.out_folder, self.ann_file), 'w') as fp:
                json.dump(annotations, fp)

            gdf_subset.to_pickle(os.path.join(path, self.out_folder, "AIS_data.pkl"))

            fig.savefig(os.path.join(path, self.out_folder, "preview_{}.png".format(image_id)))
            plt.close(fig)

            print("\x1b[31mAnnotation data successfully stored!\x1b[0m")

    def process_planet_image(self, visual_filename, mask_filename, json_filename, df_ais):
        """

        :param visual_filename: Path to the satellite imagery
        :param mask_filename: Path to the cloud mask (udm file)
        :param json_filename: Path to the annotation file (from Planet)
        :param df_ais: AIS data for this month (filtered already by ship length and status)
        :return: Patches (2x length), ratios (cloud coverage), debug figure, x and y coordinates, MMSI numbers, lengths,
                 widths, ship status, GeoPandas subset,

        """
        ##
        # LOAD THE DATA
        ##
        # Open image with gdal
        ds = gdal.Open(visual_filename)
        xoff, a, b, yoff, d, e = ds.GetGeoTransform()

        # Get projection information from source image
        ds_proj = ds.GetProjectionRef()
        ds_srs = osr.SpatialReference(ds_proj)

        # Get the source image's geographic coordinate system (the 'GEOGCS' node of ds_srs)
        geogcs = ds_srs.CloneGeogCS()

        ais_src = osr.SpatialReference()
        ais_src.SetWellKnownGeogCS("WGS84")
        transform = osr.CoordinateTransformation(ais_src, ds_srs)

        transform_tiff = osr.CoordinateTransformation(ds_srs, ais_src)
        lrx = xoff + (ds.RasterXSize * a)
        lry = yoff + (ds.RasterYSize * e)
        lon1, lat1, _ = transform_tiff.TransformPoint(xoff, yoff)
        lon2, lat2, _ = transform_tiff.TransformPoint(lrx, lry)

        try:
            with rasterio.open(visual_filename) as src:
                rgb = src.read([1, 2, 3])
        except rasterio.errors.RasterioIOError:
            print("\x1b[31mProblem while reading visual file!!\x1b[0m")
            return None

        try:
            with rasterio.open(mask_filename) as src:
                mask = src.read()
        except rasterio.errors.RasterioIOError:
            print("\x1b[31mProblem while reading mask/cloud file!!\x1b[0m")
            return None

        with open(json_filename, "r") as src:
            metadata = json.load(src)

        ##
        # PREPARE CLOUD MASK
        ##
        print("Started processing cloud mask...")
        mask_cloud = self.udm_to_mask_cloud(mask)
        mask_cloud_eroded = binary_opening(mask_cloud, structure=np.ones((1, 20, 20)).astype(mask_cloud.dtype))
        mask_cloud_eroded = binary_closing(mask_cloud_eroded, structure=np.ones((1, 10, 10)).astype(mask_cloud.dtype))
        mask_cloud_free = rgb * np.repeat(~mask_cloud_eroded, 3, axis=0)
        print("Cloud mask successfully processed!")

        ##
        # FILTER AIS DATA BY TIME
        # Convert also to GeoPandas - filter by bbox and aggregate the measurements
        ##
        start_time = (parser.parse(metadata["properties"]["acquired"]) - datetime.timedelta(minutes=self.ais_time_window)).replace(
            tzinfo=None)
        end_time = (parser.parse(metadata["properties"]["acquired"]) + datetime.timedelta(minutes=self.ais_time_window)).replace(
            tzinfo=None)
        print("Time frame for AIS merging: {} - {}".format(start_time, end_time))
        df_subset = df_ais[(df_ais["BaseDateTime"] > start_time) & (df_ais["BaseDateTime"] < end_time)]

        print("Number of AIS entries for given time frame: ", len(df_subset))
        # no AIS entries found in this timeframe
        if len(df_subset) == 0:
            print("\x1b[31mNO AIS entries found in given time frame!\x1b[0m")
            return None

        # creating a geometry column
        geometry = [Point(xy) for xy in zip(df_subset["LON"], df_subset["LAT"])]
        # Coordinate reference system : WGS84
        crs = {"init": "epsg:4326"}
        # Creating a Geographic data frame
        gdf = gpd.GeoDataFrame(df_subset, crs=crs, geometry=geometry)
        # filter by bbox
        gdf_subset = gdf.cx[lon1:lon2, lat1:lat2]

        print("Number of AIS entries for given area: ", len(gdf_subset))
        # no AIS entries found in this area
        if len(gdf_subset) == 0:
            print("\x1b[31mNO AIS entries found in given area!\x1b[0m")
            return None

        # compute mean value for location data (multiple measurements)
        gdf_subset_mean = gdf_subset[["MMSI", "LAT", "LON", "Length", "Width", "VesselType"]].groupby(["MMSI"]).mean()

        x, y = self.transform_lat_lng(gdf_subset_mean["LON"].values, gdf_subset_mean["LAT"].values, transform,
                                      xoff, yoff, a, e)
        patches, ratios, colors, patches_real = self.prepare_rects(x, y, gdf_subset_mean["Length"], float(metadata["properties"]["gsd"]),
                                                                   rgb, mask_cloud_eroded, self.cloud_thresh)

        n_valid = np.sum(np.array(ratios) < self.cloud_thresh)
        self.n_valid_planet += n_valid
        self.n_all_planet += len(x)
        print("Successfully processed {}/{} valid AIS matches (cloud cover < {}%)!".format(n_valid, len(x), int(self.cloud_thresh * 100)))
        print("\x1b[31mTotal valid matches: {} from {} all\x1b[0m".format(self.n_valid_planet, self.n_all_planet))

        fig, ax = plt.subplots(figsize=(100, 100))
        show(mask_cloud_free, ax=ax)
        ax.plot(x, y, "r+")
        coll = matplotlib.collections.PatchCollection(patches, facecolor=colors, edgecolor="red", alpha=0.5)
        ax.add_collection(coll)

        if self.debug:
            plt.show()

        return patches_real, ratios, fig, x, y, gdf_subset_mean.index, gdf_subset_mean["Length"],\
               gdf_subset_mean["Width"], gdf_subset_mean["VesselType"], gdf_subset

    @staticmethod
    def udm_to_mask_cloud(udm_array):
        """
        https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/udm/udm.ipynb
        :param udm_array: UDM tif file as provided by Planet
        :return: Cloud mask and invalid pixels
        """
        test_bits = int('00000011', 2)
        bit_matches = udm_array & test_bits

        return bit_matches != 0

    @staticmethod
    def transform_lat_lng(lon, lat, transform, xoff, yoff, a, e):
        """
        Transforms WGS84 coordinates to image coordinates
        :param lon:
        :param lat:
        :param transform: CoordinateTransformation object (AIS -> image)
        :param xoff: from GetGeoTransform
        :param yoff: from GetGeoTransform
        :param a: from GetGeoTransform
        :param e: from GetGeoTransform
        :return: x, y image coordinates
        """
        x, y = [], []
        n_coord = len(lon)
        for i in range(n_coord):
            proj_x, proj_y, _ = transform.TransformPoint(lon[i], lat[i])
            x.append((int)(((proj_x - xoff) / a)))
            y.append((int)(((proj_y - yoff) / e)))

        return np.array(x), np.array(y)

    @staticmethod
    def prepare_rects(x, y, lengths, gsd, img, cloud_mask, thresh):
        """
        Get patches for each ship, compute cloud coverage and prepare object for debug figure
        :param x:
        :param y:
        :param lengths:
        :param gsd: GSD value
        :param img: Original image (without clouds)
        :param cloud_mask: Cloud mask
        :param thresh: Cloud thresh - provided in the constructor
        :return: Patches for plotting, ratios (cloud_coverage), colors, real patches
        """
        patches = []
        colors = []
        patches_real = []
        ratios = []
        for c_x, c_y, length in zip(x, y, lengths):
            length_pix = int(length / gsd)
            patch = img[:, max(0, c_y - length_pix):min(c_y + length_pix, cloud_mask.shape[1]),
                        max(0, c_x - length_pix):min(c_x + length_pix, cloud_mask.shape[2])]
            patch_mask = cloud_mask[:, max(0, c_y - length_pix):min(c_y + length_pix, cloud_mask.shape[1]),
                                    max(0, c_x - length_pix):min(c_x + length_pix, cloud_mask.shape[2])].astype(int)
            ratio = patch_mask.flatten().sum() / (patch_mask.shape[1] * patch_mask.shape[2])
            if math.isnan(ratio):
                print("Invalid value of cloud cover - {}!".format(ratio))
                continue
            ratios.append(patch_mask.flatten().sum() / (patch_mask.shape[1] * patch_mask.shape[2]))
            patches.append(plt.Rectangle((c_x - length_pix, c_y - length_pix), length_pix * 2, length_pix * 2))
            colors.append("yellow" if ratios[-1] < thresh else "red")
            patches_real.append(patch)

        return patches, ratios, colors, patches_real

    def process_sentinel_folder(self, path, df_ais):
        """
        This method iterates through the content of each folder and stores all the annotations for this folder
        as gathered by a specific method (i.e. process_sentinel_image) to process each satellite imagery.
        :param path: Path to the folder of interest to annotate
        :param df_ais: AIS data for this month (filtered already by ship length and status)
        :return: /
        """
        annotations = []
        for eopatch in glob(os.path.join(path, "*/")):
            eopatch_id = os.path.basename(os.path.dirname(eopatch))

            if not eopatch_id.startswith("eopatch"):
                continue

            print(5 * "=")
            print("Started processing Sentinel imagery with ID: " + eopatch_id)

            satellite_imagery = "{}.tif".format(eopatch_id[2:])
            visual_filename = os.path.join(path, satellite_imagery)
            print("\x1b[31mSatellite imagery: \x1b[0m" + visual_filename)

            return_values = self.process_sentinel_image(path, eopatch_id, visual_filename, df_ais)

            if not return_values:
                print("\x1b[31mSkipping ID: {}!\x1b[0m".format(eopatch_id))
                continue
            else:
                patches_real, ratios, fig, x, y, mmsi, lengths, widths, vessel_types, gdf_subset = return_values

            if len(ratios) == 0:
                continue

            annotation = {}
            out_path_patches_valid = os.path.join(path, self.out_folder, "patches_valid")
            out_path_patches_cloudy = os.path.join(path, self.out_folder, "patches_cloudy")

            if not os.path.exists(out_path_patches_valid):
                os.makedirs(out_path_patches_valid)

            if not os.path.exists(out_path_patches_cloudy):
                os.makedirs(out_path_patches_cloudy)

            for i in range(len(ratios)):

                if math.isnan(ratios[i]):
                    print("Invalid value of cloud cover - {}!".format(ratios[i]))
                    continue

                annotation[str(mmsi[i])] = {
                    "x": int(x[i]),
                    "y": int(y[i]),
                    "cloud_cover": ratios[i],
                    "length": lengths[mmsi[i]],
                    "width": widths[mmsi[i]],
                    "vessel_type": vessel_types[mmsi[i]]
                }
                patch = np.transpose(patches_real[i], [1, 2, 0])
                patch_path = os.path.join(out_path_patches_valid if ratios[i] < self.cloud_thresh else
                                          out_path_patches_cloudy, "{}_{}_{}_{}.png".format(os.path.basename(path), eopatch_id, mmsi[i],
                                                                                         int(ratios[i] * 100)))
                imsave(patch_path, patch)

            annotations.append(annotation)
            with open(os.path.join(path, self.out_folder, self.ann_file), 'w') as fp:
                json.dump(annotations, fp)

            gdf_subset.to_pickle(os.path.join(path, self.out_folder, "AIS_data.pkl"))

            fig.savefig(os.path.join(path, self.out_folder, "preview_{}.png".format(eopatch_id)))
            plt.close(fig)

            print("\x1b[31mAnnotation data successfully stored!\x1b[0m")

    def process_sentinel_image(self, path, eopatch_id, visual_filename, df_ais):
        """
        :param path: Path to the directory where the data for date X is stored
        :param eopatch_id: eopatch name (e.g. eopatch_0)
        :param visual_filename: Path to the satellite imagery
        :param df_ais: AIS data for this month (filtered already by ship length and status)
        :return: Patches (2x length), ratios (cloud coverage), debug figure, x and y coordinates, MMSI numbers, lengths,
                 widths, ship status, GeoPandas subset,
        """
        ##
        # LOAD THE DATA
        ##
        # Open image with gdal
        ds = gdal.Open(visual_filename)
        xoff, a, b, yoff, d, e = ds.GetGeoTransform()

        # Get projection information from source image
        ds_proj = ds.GetProjectionRef()
        ds_srs = osr.SpatialReference(ds_proj)

        # Get the source image's geographic coordinate system (the 'GEOGCS' node of ds_srs)
        geogcs = ds_srs.CloneGeogCS()

        ais_src = osr.SpatialReference()
        ais_src.SetWellKnownGeogCS("WGS84")
        transform = osr.CoordinateTransformation(ais_src, ds_srs)

        transform_tiff = osr.CoordinateTransformation(ds_srs, ais_src)
        lrx = xoff + (ds.RasterXSize * a)
        lry = yoff + (ds.RasterYSize * e)
        lon1, lat1, _ = transform_tiff.TransformPoint(xoff, yoff)
        lon2, lat2, _ = transform_tiff.TransformPoint(lrx, lry)

        ##
        # LOAD EOPatch
        ##
        patch_load_task = LoadFromDisk(path)
        eopatch = patch_load_task.execute(eopatch_folder=eopatch_id)
        rgb = np.transpose(eopatch["data"]["TRUE-COLOR-S2-L1C"], (0, 3, 1, 2))[0]
        mask_cloud = np.logical_or(eopatch["mask"]["CLM"].astype("bool"), ~eopatch["mask"]["IS_DATA"].astype("bool"))
        mask_cloud = np.transpose(mask_cloud, (0, 3, 1, 2))[0]
        print("EOPatch successfully loaded!")

        ##
        # PREPARE CLOUD MASK
        ##
        print("Started processing cloud mask...")
        mask_cloud_eroded = binary_opening(mask_cloud, structure=np.ones((1, 20, 20)).astype(mask_cloud.dtype))
        mask_cloud_eroded = binary_closing(mask_cloud_eroded, structure=np.ones((1, 10, 10)).astype(mask_cloud.dtype))
        mask_cloud_free = rgb * np.repeat(~mask_cloud_eroded, 3, axis=0)
        print("Cloud mask successfully processed!")

        ##
        # FILTER AIS DATA BY TIME
        # Convert also to GeoPandas - filter by bbox and aggregate the measurements
        ##

        start_time = (eopatch["timestamp"][0] - datetime.timedelta(
            minutes=self.ais_time_window)).replace(
            tzinfo=None)
        end_time = (eopatch["timestamp"][0] + datetime.timedelta(
            minutes=self.ais_time_window)).replace(
            tzinfo=None)
        print("Time frame for AIS merging: {} - {}".format(start_time, end_time))
        df_subset = df_ais[(df_ais["BaseDateTime"] > start_time) & (df_ais["BaseDateTime"] < end_time)]

        print("Number of AIS entries for given time frame: ", len(df_subset))
        # no AIS entries found in this timeframe
        if len(df_subset) == 0:
            print("\x1b[31mNO AIS entries found in given time frame!\x1b[0m")
            return None

        # creating a geometry column
        geometry = [Point(xy) for xy in zip(df_subset["LON"], df_subset["LAT"])]
        # Coordinate reference system : WGS84
        crs = {"init": "epsg:4326"}
        # Creating a Geographic data frame
        gdf = gpd.GeoDataFrame(df_subset, crs=crs, geometry=geometry)
        # filter by bbox
        gdf_subset = gdf.cx[lon1:lon2, lat1:lat2]

        print("Number of AIS entries for given area: ", len(gdf_subset))
        # no AIS entries found in this area
        if len(gdf_subset) == 0:
            print("\x1b[31mNO AIS entries found in given area!\x1b[0m")
            return None

        # compute mean value for location data (multiple measurements)
        gdf_subset_mean = gdf_subset[["MMSI", "LAT", "LON", "Length", "Width", "VesselType"]].groupby(["MMSI"]).mean()

        x, y = self.transform_lat_lng(gdf_subset_mean["LON"].values, gdf_subset_mean["LAT"].values, transform,
                                      xoff, yoff, a, e)
        patches, ratios, colors, patches_real = self.prepare_rects(x, y, gdf_subset_mean["Length"], 10.0,
                                                                   rgb, mask_cloud_eroded, self.cloud_thresh)

        n_valid = np.sum(np.array(ratios) < self.cloud_thresh)
        self.n_valid_planet += n_valid
        self.n_all_planet += len(x)
        print("Successfully processed {}/{} valid AIS matches (cloud cover < {}%)!".format(n_valid, len(x), int(
            self.cloud_thresh * 100)))
        print("\x1b[31mTotal valid matches: {} from {} all\x1b[0m".format(self.n_valid_planet, self.n_all_planet))

        fig, ax = plt.subplots(figsize=(100, 100))
        show(mask_cloud_free, ax=ax)
        ax.plot(x, y, "r+")
        coll = matplotlib.collections.PatchCollection(patches, facecolor=colors, edgecolor="red", alpha=0.5)
        ax.add_collection(coll)

        if self.debug:
            plt.show()

        return patches_real, ratios, fig, x, y, gdf_subset_mean.index, gdf_subset_mean["Length"], \
               gdf_subset_mean["Width"], gdf_subset_mean["VesselType"], gdf_subset

    def load_AIS(self, date, location, max_length):
        """

        :param date: date to extract month from (AIS data is stored by month)
        :param location: planet_data or sf_bay to load either Zone10 or Zone11
        :param max_length: Max length of the ship -> from the constructor
        :return: Pandas DataFrame
        """
        ais_filename = os.path.join(self.ais_dirname, str(date.year), "AIS_{:%Y_%m}_{}".format(date, "Zone10.csv"
                                    if location == "sf_bay" else "Zone11.csv"))
        print("\x1b[31mAIS data: \x1b[0m" + ais_filename)
        df = pd.read_csv(ais_filename, low_memory=False)
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
        df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
        df = df.loc[df["Length"] > max_length]
        df = df[df["Status"] != "under way using engine"]

        return df

    def delete_empty(self, start_date, end_date, providers=["planet_data", "sentinel_data"],
                      locations=["long_beach", "sf_bay"]):

        """
        This method will clear empty directories in the dataset
        :param start_date: Start date for annotations
        :param end_date: End date for annotations
        :param providers: List of providers (currently: planet_data and sentinel_data)
        :param locations: List of locations (currently: long_beach and sf_bay)
        :return: /
        """
        for provider in providers:
            for location in locations:
                for date in pd.date_range(start_date, end_date):
                    path_curr = os.path.join(self.dataset_dir, provider, location, "{:%Y_%m_%d}".format(date))
                    if os.path.exists(path_curr):
                        try:
                            os.rmdir(path_curr)
                            print("Removed: " + path_curr)
                        except OSError:
                            continue

    def print_stats(self, start_date, end_date, providers=["planet_data", "sentinel_data"],
                      locations=["long_beach", "sf_bay"], debug=True):
        """
            This method will print dataset statistics
            :param start_date: Start date for annotations
            :param end_date: End date for annotations
            :param providers: List of providers (currently: planet_data and sentinel_data)
            :param locations: List of locations (currently: long_beach and sf_bay)
            :param debug: if True, print results
            :return: Dictionary with valid/fail counter for each location
        """
        stats = {
            "SF_valid": 0,
            "SF_fail": 0,
            "LB_valid": 0,
            "LB_fail": 0
        }

        for provider in providers:
            for location in locations:
                for date in pd.date_range(start_date, end_date):
                    path_curr = os.path.join(self.dataset_dir, provider, location, "{:%Y_%m_%d}".format(date), self.out_folder)

                    # check if there is annotation folder
                    if os.path.exists(path_curr):
                        path_ann_0 = os.path.join(path_curr, "patches_cloudy")
                        path_ann_1 = os.path.join(path_curr, "patches_valid")

                        if os.path.exists(path_ann_0):
                            if location == "long_beach":
                                stats["LB_fail"] += len(glob(os.path.join(path_ann_0, "*.png")))
                            elif location == "sf_bay":
                                stats["SF_fail"] += len(glob(os.path.join(path_ann_0, "*.png")))

                        if os.path.exists(path_ann_1):
                            if location == "long_beach":
                                stats["LB_valid"] += len(glob(os.path.join(path_ann_1, "*.png")))
                            elif location == "sf_bay":
                                stats["SF_valid"] += len(glob(os.path.join(path_ann_1, "*.png")))

        if debug:
            print("SF valid samples: ", stats["SF_valid"])
            print("SF cloudy samples: ", stats["SF_fail"])
            print(10 * "=")
            print("LB valid samples: ", stats["LB_valid"])
            print("LB cloudy samples: ", stats["LB_fail"])

        return stats


if __name__ == '__main__':
	pass
