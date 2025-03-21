
import os
from pathlib import Path

import numpy as np
import geopandas as gpd
from dotenv import load_dotenv
from tqdm import tqdm
from rasterio.merge import merge
import rasterio as rio
from datetime import datetime


import utils


def correctTiles(folder):
    '''
    Only use that tiles that contain pixels
    :param folder:
    :return:
    '''
    folder = Path(folder)
    load_dotenv()
    outputfolder = Path(os.environ['workdirectory'])
    # get tile file LabelersTiles
    tilefile = outputfolder / 'Tiles.shp'
    # tilefile = os.path.join(folder, 'LabelersTiles.shp')
    df = gpd.read_file(tilefile)
    df['bEmpty'] = False

    # get all tif files & open them
    # tifs = utils.getTifFiles(folder)
    tifs = [
        folder / str(os.environ['name_ortho_stitch'])
    ]
    sources = [rio.open(f) for f in tifs]

    # loop over all tiles & tag the empty ones
    l = len(df)
    for idx, tiledf in tqdm(df.iterrows(), desc='Removing empty boxes', unit='box', total=len(df)):
        poly = tiledf.geometry
        #polydf = gpd.GeoDataFrame(gpd.GeoSeries([poly]), columns=['geometry'])
        # extract RGB tile
        tile, out_trans = merge(sources, bounds=poly.bounds, precision=15)

        tile = tile.swapaxes(0,1).swapaxes(1,2)

        if(np.max(tile) == 0):
            df.loc[idx, 'bEmpty'] = True
        elif(np.max(tile) == -32767): # Own set "no data" value
            df.loc[idx, 'bEmpty'] = True

    nonEmptyDf = df[df['bEmpty'] == False]

    #outfile = os.path.join(folder, 'NonEmptyTiles.shp') LabelersTiles
    outfile = outputfolder / 'Tiles.shp'
    outfile.parent.mkdir(exist_ok=True, parents=True)
    nonEmptyDf.to_file(outfile)

    # and we make a backup
    outfile = outputfolder / 'backup' / 'Tiles.shp'
    outfile.parent.mkdir(exist_ok=True, parents=True)
    outfile = str(outfile).replace('.shp', datetime.now().strftime('__%d%m%Y_%H%M.shp'))
    df.to_file(outfile)

if __name__ == '__main__':

    load_dotenv()
    correctTiles(os.environ['orthofolder'])





