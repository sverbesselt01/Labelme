
import os
import os.path as osp
from pathlib import Path
from typing import Generator

import cv2
import copy
import json
import warnings

from geopandas import GeoDataFrame
from skimage.transform import resize

import rasterio as rio
from rasterio.merge import merge
from rasterio.transform import rowcol
from rasterio import Affine
from shapely.geometry import Polygon, Point

import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
from sklearn.model_selection import train_test_split

from image import rescale_images, preprocess_vis_image
import config, utils

warnings.filterwarnings('once', message='GeoSeries crs mismatch')

class TileGen(Generator):
    def __init__(self, imgSize, datafolder,
                 tiffiles=[],
                 tileFiles=['Tiles.shp'], labelFiles=['Labels.shp'], batchSize=1,
                 buffer=0, ssf=1,
                 bEmpty=False, transformParams=None, preprocessParams=None,
                 **kwargs):
        if bEmpty:
            # don't do anything, just return, the object will be filled later
            return

        kwargs['image_min_side'] = np.min(imgSize)//ssf
        kwargs['image_max_side'] = np.max(imgSize)//ssf
        kwargs['batch_size'] = batchSize

        self.imgSize = imgSize
        self.buffer = buffer # adjust to 7
        self.batchSize = batchSize
        self.datafolder = datafolder
        self.ssf = ssf # subsample factor

        self.tiffiles = tiffiles
        self.tileFiles = tileFiles
        for i, tf in enumerate(self.tileFiles):
            if osp.split(tf)[-1] == tf:
                self.tileFiles[i] = osp.join(datafolder, tf)

        self.labelFiles = labelFiles
        for i, lf in enumerate(self.labelFiles):
            if osp.split(lf)[-1] == lf:
                self.labelFiles[i] = osp.join(datafolder, lf)

        # 0. load & prepare tifs
        # 1. read labelfile
        #   a. import label names
        #   b. load geometry -> bboxes
        # 2. read Tiles
        # 3. save tiles to tmp file -> ready for loading
        self.loadData()

        # self.names is defined in loadLabelfile(), by extracting all unique QGis labels
        self.classes = {n:i for i, n in enumerate(self.names)}
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.kwargs = kwargs
        self.preprocessParams = preprocessParams
        self.transformParams = transformParams
        super(TileGen, self).__init__(**kwargs)

    def __len__(self):
        return len(self.groups)

    def __repr__(self):
        ret = 'TileGen with {} samples in {} batches and \n' \
              'classes: {}'.format(self.size(), len(self), ', '.join(self.names))
        return ret

    def size(self):
        """ Size of the dataset.
        """
        return len(self.lTiledf)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def has_label(self, label):
        return (label in self.labels)

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return self.lTiledf.loc[image_index, 'filepath']

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        tile = rio.open(self.lTiledf.loc[image_index, 'filepath'])
        return float(tile.width/tile.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        tile = rio.open(self.lTiledf.loc[image_index, 'filepath'])
        image = tile.read(out_shape=(4, tile.height//self.ssf, tile.width//self.ssf))
        image = image.swapaxes(0,1).swapaxes(1,2)
        #utils.log('Image:', array=image)
        #image = (image/255.).astype(np.float32)
        return image[..., :3]

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        labels = self.lTiledf.loc[image_index, 'labels']
        polys = self.lTiledf.loc[image_index, 'bboxes']
        tile = rio.open(self.lTiledf.loc[image_index, 'filepath'])

        # loop over bboxes (=POLYGONS)
        for idx in range(len(labels)):
            poly = polys[idx]
            x1, y1, x2, y2 = poly.bounds
            xs, ys = [x1, x2], [y1, y2]
            rows, cols = rowcol(tile.transform, xs, ys)
            # enlarge bounding boxes
            f = 0
            rows = [min(rows) * (1-f), max(rows) * (1+f)]
            cols = [min(cols) * (1-f), max(cols) * (1+f)]
            # clip, otherwise bboxes get removed by RetinaNet
            rows = np.clip(rows, a_min=0, a_max=self.imgSize[0])
            cols = np.clip(cols, a_min=0, a_max=self.imgSize[1])
            box = np.array(list(zip(cols, rows))) #rows & cols needs be switched for retinanet
            box = box.flatten()//self.ssf

            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(labels[idx])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [box]))

        return annotations

    def loadTifs(self):
        # get all tif files & open them
        self.sources = [rio.open(f) for f in self.tiffiles]

    def loadLabelfile(self, labelfiles=None):
        if labelfiles is None:
            labelfiles = [os.path.join(self.datafolder, f) for f in os.listdir(self.datafolder) if
                          f.startswith('Labels') and f.endswith('.shp')]

        #labelfiles = sorted(labelfiles, key=utils.sortLabelFiles)
        #labelfile = labelfiles[-1]

        self.labeldf = None
        for labelfile in labelfiles:
            print('Reading labelfile: {}'.format(labelfile))
            df = gpd.read_file(labelfile)
            if self.labeldf is None:
                self.labeldf = df
            else:
                self.labeldf = gpd.GeoDataFrame(pd.concat([self.labeldf, df], ignore_index=True), crs=self.labeldf.crs)

        self.labeldf = self.labeldf[~self.labeldf.geometry.isnull()]

        self.names = self.labeldf['Label'].unique()
        self.names = np.insert(self.names, 0, 'bg')

    def loadTilefile(self):
        # get tile file
        tiledf = None
        for tileFile in self.tileFiles:
            df = gpd.read_file(tileFile)
            if tiledf is None:
                tiledf = df
            else:
                tiledf = gpd.GeoDataFrame(pd.concat([tiledf, df], ignore_index=True), crs=tiledf.crs)

        tiledf['filepath'] = ""
        tiledf['labels'] = np.empty((len(tiledf), 0)).tolist()
        tiledf['bboxes'] = np.empty((len(tiledf), 0)).tolist()

        self.lTiledf = tiledf[tiledf['bTileUsed'] == True].copy()  # labelled
        self.nlTiledf = tiledf[tiledf['bTileUsed'] == False].copy() # non-labelled

        self.lTiledf.reset_index(inplace=True)
        self.nlTiledf.reset_index(inplace=True)

        self.lTiledf['nLabels'] = 0
        for idx, tiledf in self.lTiledf.iterrows():
            poly = tiledf.geometry
            polydf = gpd.GeoDataFrame(gpd.GeoSeries([poly]), columns=['geometry'], crs=self.lTiledf.crs)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                tmplabels = gpd.overlay(self.labeldf, polydf, how='intersection')

            self.lTiledf.loc[idx, 'nLabels'] = len(tmplabels)
            #if(len(tmplabels) == 0):
            #    continue

            # extract RGB tile
            tile, out_trans = merge(self.sources, bounds=poly.bounds)
            # save & reload file to convert the ndarray to rasterio dataset
            out_meta = self.sources[0].meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": tile.shape[1],
                             "width": tile.shape[2],
                             "transform": out_trans
                             })
            # 1) make temp file
            tmpfile = tempfile.mktemp('.tif')
            # 2) write to tempfile
            with rio.open(tmpfile, "w", **out_meta) as dest:
                dest.write(tile)
            # 3) save reference
            self.lTiledf.loc[idx, 'filepath'] = tmpfile
            # 4) add tmplabels
            self.lTiledf.loc[idx, 'labels'].extend(list(tmplabels['Label']))
            self.lTiledf.loc[idx, 'bboxes'].extend(list(tmplabels.geometry))

        #self.lTiledf = self.lTiledf[self.lTiledf['nLabels'] > 0]
        #self.lTiledf.reset_index(inplace=True, drop=True)

    def loadData(self, idx=None):
        self.loadTifs()
        self.loadLabelfile(self.labelFiles)
        self.loadTilefile()

    def getSplit(self, testfrac=0.2):
        trainidx, testidx = train_test_split(self.lTiledf.index, test_size=testfrac)

        return trainidx, testidx

    def split(self, idx):
        # make empty generators
        newGen = type(self)([], '', bEmpty=True)

        attrs = ['imgSize', 'buffer', 'batchSize', 'datafolder', 'ssf', 'tileFiles', 'labelFiles', 'kwargs',
                 'classes', 'labels', 'names',
                 'labeldf', 'lTiledf', 'nlTiledf']

        for attr in attrs:
            val = self.__getattribute__(attr)
            bCopied = False
            cp = getattr(val, 'copy', None)
            if(cp is not None):
                if callable(cp):
                    val = val.copy()
                    bCopied = True

            if not bCopied:
                val = copy.copy(val)

            newGen.__setattr__(attr, val)

        # load other files to ensure compatibility
        # get all tif files & open them
        tifs = utils.getTifFiles(newGen.datafolder)
        newGen.sources = [rio.open(f) for f in tifs]

        # perform split
        msk = self.lTiledf.index.isin(idx)
        newGen.lTiledf = newGen.lTiledf[msk].reset_index(drop=True)
        super(TileGen, newGen).__init__(**self.kwargs)

        return newGen

    def replaceLabels(self, old_to_new):
        def replaceInList(lst, oldNew):
            newlst = [oldNew.get(l, l) for l in lst]

        self.labeldf['Label'] = self.labeldf['Label'].replace(old_to_new)
        # change entrys in list, if possible, if not in dictionary, retain list value
        self.lTiledf['labels'] = self.lTiledf['labels'].apply(lambda lst: [old_to_new.get(l, l) for l in lst])

        self.names = self.labeldf['Label'].unique()
        self.names = np.insert(self.names, 0, 'bg')

        # self.names is defined in loadLabelfile(), by extracting all unique QGis labels
        self.classes = {n:i for i, n in enumerate(self.names)}
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def save(self, folder):
        '''
        Save data generator to speed-up the process
        :param filename:
        :return:
        '''

        if not osp.exists(folder):
            os.makedirs(folder)

        # save parameters & dataframes
        attrs = ['imgSize', 'buffer', 'batchSize', 'datafolder', 'ssf', 'tileFiles', 'labelFiles', 'kwargs',
                 'classes', 'labels', 'names', 'transformParams', 'preprocessParams',
                 'labeldf', 'lTiledf', 'nlTiledf']

        parameters = {}
        for attr in attrs:
            val = self.__getattribute__(attr)
            if type(val) == pd.DataFrame or type(val) == gpd.GeoDataFrame:
                outfile = osp.join(folder, '{}.pkl'.format(attr))
                val.to_pickle(outfile)
            # elif type(val) == gpd.GeoDataFrame:
            #     outfile = osp.join(folder, '{}.shp'.format(attr))
            #     for col in val.columns:
            #         if type(val[col][0]) == 'list':
            #             val[col] = val[col].apply(lambda lst: 'list:' + ','.join(lst))
            #         print('Column {} has dtype: {}, the first item is of type {} (value: {})'.format(col, val[col].dtype, type(val[col][0]), val[col][0]))
            #
            #     val.to_file(outfile)
            elif type(val) == dict:
                tmpdict = {}
                # filter all subkeys that are not serializable
                for key, va in val.items():
                    try:
                        # test if we can serialize the parameter
                        json.dumps({key: va})
                    except TypeError:
                        # if not, skip
                        continue
                    tmpdict[key] = va

                parameters[attr] = tmpdict
            else:
                print('{}: {}'.format(attr, val))
                try:
                    # test if we can serialize the parameter
                    json.dumps({attr:val})
                except TypeError:
                    # if not, skip
                    continue
                # if yes, store it
                parameters[attr] = val

        outfile = osp.join(folder, 'parameters.json')
        with open(outfile, 'w') as f:
            json.dump(parameters, f)

    def load(self, folder):

        outfile = osp.join(folder, 'parameters.json')
        with open(outfile) as f:
            parameters = json.load(f)

        # set parameters
        for attr, val in parameters.items():
            setattr(self, attr, val)

        pklfiles = [osp.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl')]
        for pkl in pklfiles:
            df = pd.read_pickle(pkl)
            attr = osp.splitext(osp.split(pkl)[-1])[0]
            setattr(self, attr, df)

        shpfiles = [osp.join(folder, f) for f in os.listdir(folder) if f.endswith('.shp')]
        for shp in shpfiles:
            df = gpd.read_file(shp)
            attr = osp.splitext(osp.split(shp)[-1])[0]
            setattr(self, attr, df)

        tifs = utils.getTifFiles(self.datafolder)
        self.sources = [rio.open(f) for f in tifs]

        # # add preprocess & transform generator
        # self.kwargs['transform_generator'] = random_transform_generator(**self.transformParams)
        # self.kwargs['preprocess_image'] = rescale_images(**self.preprocessParams)
        super(TileGen, self).__init__(**self.kwargs)

#TODO: adjust for labers (!)
# So that each labeler has a seperate file
class ApplyGenerator():
    def __init__(self, datafolder,
                 tiffiles = [],
                 tileFile='Tiles.shp', labelFile='Labels.shp',
                 ssf=1, datasource='prediction', threshold=0.7):
        '''
        TODO: add explanation
        :param datafolder:
        :param tileFile:
        :param ssf:
        :param datasource:
        :param threshold: threshold above which a detection is selected
        '''
        self.labeldf = None
        self.datafolder = datafolder
        self.datasource = datasource

        self.tiffiles = tiffiles

        self.tileFile = tileFile
        if osp.split(self.tileFile)[-1] == self.tileFile:
            self.tileFile = osp.join(datafolder, self.tileFile)

        self.labelFile = labelFile
        if osp.split(self.labelFile)[-1] == self.labelFile:
            self.labelFile = osp.join(datafolder, self.labelFile)

        self.ssf = ssf
        self.threshold = threshold

        assert datasource in ['labeler', 'prediction']

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.tiledf):
            self.i += 1
            return self.tiledf.index[self.i-1]
        else:
            raise StopIteration

    # other operators? e.g. with-operator -> to open & auto close?
    def __len__(self):
        return len(self.tiledf)

    def isLabeled(self, idx):
        tdf = self.tiledf.loc[idx]
        return bool(tdf['bTileUsed'])

    def __getitem__(self, idx):
        tdf = self.tiledf.loc[idx]
        poly = tdf.geometry
        # extract RGB tile
        tile, out_trans = merge(self.sources, bounds=poly.bounds)
        tile = tile.swapaxes(0,1).swapaxes(1,2)[..., :3]

        if self.ssf > 1:
            tile = resize(tile, (tile.shape[0]//self.ssf, tile.shape[1]), anti_aliasing=True)

        return tile

    def __setitem__(self, idx, value):
        # unpack
        if len(value) == 2:
            bboxes, labels = value
            scores = [1.0 for i in range(len(labels))]
        elif len(value) == 3:
            bboxes, labels, scores = value
        else:
            warnings.warn('Warning: Setting a tile using {} requires a list/tupple '
                          'with 2 or 3 items (bboxes, labels [, scores]), got {} items.'
                          'Leaving tile untouched'.format(type(self), len(value)))
            return

        if self.ssf > 1:
            bboxes *= self.ssf

        # get affine transform
        tdf = self.tiledf.loc[idx]
        bounds = tdf.geometry.bounds
        #GeoDataFrame(geometry=[tdf.geometry]).to_file('tmp.shp')
        transf = self.calcTransform(bounds)

        for i, geom in enumerate(bboxes):
            if len(geom) == 2 or len(geom) == 4:
                if len(geom) == 4:
                    x1 = geom[0]
                    x2 = geom[2]
                    y1 = geom[1]
                    y2 = geom[3]
                if len(geom) == 2:
                    x1 = min(geom[:, 0])
                    x2 = max(geom[:, 0])
                    y1 = min(geom[:, 1])
                    y2 = max(geom[:, 1])
                poly = np.array(utils.bbox2poly(points=[(x1, y1), (x2, y2)]))
                tpoly = []
                for pt in poly:
                    tpoly.append(transf * pt)
                poly = Polygon(tpoly)
            elif len(geom) == 1:
                # TODO: at this moment, we skip the points, fix Fiona error
                poly = Point(transf * geom[0])
                continue
            else:
                continue

            polydf = gpd.GeoDataFrame(index=[0], geometry=[poly])
                                      #crs=self.sources[0].crs)
            polydf['Label'] = labels[i]
            polydf['Score'] = scores[i]
            polydf['bValid'] = scores[i] > self.threshold
            if(self.datasource == 'labeler'):
                self.tiledf.loc[idx, 'bTileUsed'] = True
                polydf['Labeler'] = tdf['Labeler'].copy()
            elif(self.datasource == 'prediction'):
                polydf['bPredicted'] = True

            if(self.labeldf is None):
                self.labeldf = polydf
            else:
                self.labeldf = pd.concat([self.labeldf, polydf])

    def __repr__(self):
        return 'ApplyGenerator with {} tiles'.format(len(self))

    def loadTilefile(self):
        # get tile file
        self.tiledf = gpd.read_file(self.tileFile)
        # Ensure the column 'bTileUsed' is boolean
        if 'bTileUsed' in self.tiledf.columns:
           self.tiledf['bTileUsed'] = self.tiledf['bTileUsed'].astype(int).astype(bool)
        #self.tiledf.reset_index(inplace=True, drop=True)

    def calcTransform(self, bounds):
        '''
        Copied steps from rasterio.merge, except the data copying
        :return: Affine()
        '''

        dst_w, dst_s, dst_e, dst_n = bounds
        output_transform = Affine.translation(dst_w, dst_n)

        first = self.sources[0]
        first_res = first.res

        output_transform *= Affine.scale(first_res[0], -first_res[1])

        return output_transform

    def saveLabels(self):
        # write outfile
        labelfiles = [os.path.join(self.datafolder, f) for f in os.listdir(self.datafolder) if
                      f.startswith('Labels') and f.endswith('.shp')]
        labelfiles = sorted(labelfiles, key=utils.sortLabelFiles)
        version = len(labelfiles)

        if self.labeldf is None:
            warnings.warn('Warning: Empty dataframe with labels will not be saved.')
        else:
            if self.datasource == 'prediction':
                self.labelFile = os.path.join(self.datafolder, 'Labels_v{}.shp'.format(version))
            elif self.datasource == 'labeler':
                labeler = list(self.labeldf['Labeler'].mode())[0]
                self.labelFile = os.path.join(self.datafolder, 'Labeler{}.shp'.format(labeler))
                self.tiledf.to_file(self.tileFile)

            self.labeldf.crs = self.sources[0].crs
            self.labeldf.to_file(self.labelFile)

    def open(self):
        # get all tif files & open them
        self.sources = [rio.open(f) for f in self.tiffiles]

        self.loadTilefile()

    def close(self):
        self.saveLabels()

        for src in self.sources:
            src.close()

    def selectByUUID(self, uuid):
        seldf = self.tiledf[self.tiledf['TileID'] == uuid]
        try:
            #idx = list(seldf.index)
            idx = list(seldf.index)[0]
            return idx
        except IndexError:
            raise ValueError('Entry with UUID \'{}\' not found in database'.format(uuid))

if __name__ == '__main__':
    leftkeys = (81, 110, 65361, 2424832)
    rightkeys = (83, 109, 65363, 2555904)

    folder = Path(os.environ['orthofolder'])
    batchSize = config.batchSize
    subSampleFactor = config.ssf
    tilesize = int(os.environ['tilesize'])
    transformParams = {
        'min_rotation': -0.1,
        'max_rotation': 0.1,
        'min_translation': (-0.1, -0.1),
        'max_translation': (0.1, 0.1),
        'min_shear': -0.1,
        'max_shear': 0.1,
        'min_scaling': (0.9, 0.9),
        'max_scaling': (1.1, 1.1),
        'flip_x_chance': 0.5,
        'flip_y_chance': 0.5
    }
    vizParams = {
        'contrast_range': (0.9, 1.1),
        'brightness_range': (-.1, .1),
        'hue_range': (-0.05, 0.05),
        'saturation_range': (0.95, 1.05)
    }
    ap = AnchorParameters(**config.anchorParams)

    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(**transformParams)

    traingen = TileGen(imgSize=tilesize, datafolder=folder, batchSize=batchSize, ssf=subSampleFactor,
                       transform_generator=transform_generator,
                       preprocess_image=preprocess_vis_image(**vizParams)) #lambda x: (x/255.).astype(np.float32)

    print('Images in generator: {}'.format(traingen.size()))
    print('Classes in generator: {}\t\t=> {}'.format(len(traingen.names), traingen.names))

    # display images, one at a time
    bTransform = True
    bResize = True
    bDrawAnchors = True
    bDrawAnnotations = True
    bDrawCaption = True
    i = 0
    while True:
        # load the data
        image       = traingen.load_image(i)
        annotations = traingen.load_annotations(i)

        anchors = anchors_for_shape(image.shape, anchor_params=ap)
        utils.log('Anchors:', array=anchors)

        if len(annotations['labels']) > 0 :
            # apply random transformations
            if bTransform:
                image, annotations = traingen.random_transform_group_entry(image, annotations)
                #image, annotations = traingen.random_visual_effect_group_entry(image, annotations)
            if bResize:
                image, image_scale = traingen.resize_image(image)
                annotations['bboxes'] *= image_scale

            positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

            if bDrawAnchors:
                draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=1)

            if bDrawAnnotations:
                # draw annotations in red
                draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=traingen.label_to_name)

                # draw regressed anchors in green to override most red annotations
                # result is that annotations without anchors are red, with anchors are green
                draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))
            if bDrawCaption:
                draw_caption(image, [0, image.shape[0]], os.path.basename(traingen.image_path(i)))

        # if we are using the GUI, then show an image
        cv2.imshow('Image', image)
        key = cv2.waitKeyEx()

        # press right for next image and left for previous (linux or windows, doesn't work for macOS)
        # if you run macOS, press "n" or "m" (will also work on linux and windows)

        if key in rightkeys:
            i = (i + 1) % traingen.size()
        if key in leftkeys:
            i -= 1
            if i < 0:
                i = traingen.size() - 1

        # press q or Esc to quit
        if (key == ord('q')) or (key == 27):
            break
