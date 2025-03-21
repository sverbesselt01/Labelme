import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from dotenv import load_dotenv
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate





from utils import log, bbox2poly


def calcTiles(regionBbox, delta, buffer=5):
    '''
    splits the defined region into tiles of a certain size
    :param regionBbox:
    :param tilesize
    :return:
    '''
    (x1, y1), (x2, y2) = regionBbox
    xs = tuple((x1, x2))
    x1 = min(xs)
    x2 = max(xs)
    ys = tuple((y1, y2))
    y1 = min(ys)
    y2 = max(ys)
    #overlap = delta/2 # 50% overlap
    overlap = 0
    xs = np.arange(x1 - delta * buffer, x2 + delta * buffer, delta - overlap)
    ys = np.arange(y1 - delta * buffer, y2 + delta * buffer, delta - overlap)

    bboxes = []
    for x in xs:
        for y in ys:
            x2 = x + delta
            y2 = y + delta
            bbox = Polygon(bbox2poly([(x, y), (x2, y2)]))
            bboxes.append(bbox)

    return bboxes

if __name__ == '__main__':

    load_dotenv()
    folder = Path(os.environ['orthofolder'])
    outputfolder = Path(os.environ['workdirectory'])


    tilesize = int(os.environ['tilesize'])
    #tifs = [folder / f for f in os.listdir(folder) if f.endswith('.tif')]
    tifs = [
        folder / str(os.environ['name_ortho_stitch'])
    ]
    print('Listed {} files'.format(len(tifs)))

    points = [
        (0, 0),
        (-1, -1)
    ]
    coords = None
    deltas = None
    for file in tifs:
        with rio.open(file) as src:
            crs = src.crs
            points = [
                (0, 0),
                (src.height, src.width)
            ]
            for i, (x, y) in enumerate(points):
                pt = src.xy(x, y)
                print((x,y), ': ', pt)
                pt2 = src.xy(tilesize, tilesize)
                if coords is None:
                    coords = np.asarray([pt])
                    deltas = np.abs(np.asarray([(pt2[0]-pt[0], pt2[1] - pt[1])]))
                else:
                    if i == 0:
                        d = np.abs(np.asarray((pt2[0] - pt[0], pt2[1] - pt[1])))
                        deltas = np.append(deltas, [d], axis=0)
                    coords = np.append(coords, [pt], axis=0)

    print('CRS:', crs)


    log('Coords:', array=coords)
    log('Deltas:', array=deltas)
    mean = np.mean(deltas, axis=0)
    print('Mean deltas:', mean)

    #add extra region by x1.05
    top = min(coords[:, 0])
    left = min(coords[:, 1])
    bottom = max(coords[:, 0])
    right = max(coords[:, 1])
    print('Top left: ({}, {})'.format(top, left) )
    print('Bottom right: ({}, {})'.format(bottom, right))

    # build boxes of 512 x 512
    region = [(top, left), (bottom, right)]
    print('Region: ', region)
    bboxes = calcTiles(region, delta=np.mean(deltas), buffer=12)

    # construct geodataframe
    # bbox = Polygon(bbox2poly([(top, left), (bottom, right)]))
    if crs is not None:
        crs = {'init': str(crs)}
    bboxgdf = gpd.GeoDataFrame(index=[list(range(len(bboxes)))], crs=crs,
                               geometry=bboxes)
    bboxgdf['bTileUsed']= [False for i in range(len(bboxes))]
    bboxgdf['TileID']= [str(uuid4()) for i in range(len(bboxes))]



    # output file
    outfile = outputfolder / 'Tiles.shp'
    outfile.parent.mkdir(parents=True, exist_ok=True)
    print('Writing tiles to file {}'.format(outfile))
    bboxgdf.to_file(outfile)

    # and we make a backup
    outfile = outputfolder / 'backup' / 'Tiles.shp'
    outfile.parent.mkdir(exist_ok=True, parents=True)
    outfile = str(outfile).replace('.shp', datetime.now().strftime('__%d%m%Y_%H%M.shp'))
    bboxgdf.to_file(outfile)


