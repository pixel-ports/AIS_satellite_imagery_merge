## AIS and Satellite imagery dataset builder

###### GeoTIFF_Planet.ipynb

This notebook demonstrates AIS and Satellite imagery merging from Planet and
MarineCadastre.

More information about [Planet UDM mask](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/udm/udm.ipynb)

Notebook contains Modified Open California Satellite Imagery ©2019 Planet Labs Inc. licensed under CC BY-SA 4.0.

```
conda install -c conda-forge rasterio
conda install -c conda-forge gdal
conda install -c conda-forge libiconv (if libiconv.so.2 missing)
conda install matplotlib
conda install geopandas
conda install dask
```

Additionaly to prepare conda enviroment for the Jupyter the following is needed:

```
conda install jupyter
python -m ipykernel install --user --name CHANGE_ENV_NAME --display-name "Python (CHANGE_ENV_NAME)"
```



###### GeoTIFF_ESA.ipynb

This notebook demonstrates AIS and Satellite imagery merging from ESA Sentinel-2
and MarineCadastre.

More information about [eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html)

Notebook contains modified Copernicus Sentinel data 2019, obtained with Sentinel Hub licensed under CC BY-NC 4.0.

```
pip install eo-learn
```

Tested with:
- Python version 3.6
- Conda version 4.6.11

Conda enviroment is also exported and available in requirements directory