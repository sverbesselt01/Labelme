import os
import cv2
import numpy as np
from datetime import datetime

html_string = '''
<html>
  <head><title>HTML Pandas Dataframe with CSS</title></head>
  <link rel="stylesheet" type="text/css" href="dfstyle.css"/>
  <body>
    {table}
  </body>
</html>.
'''

def log(text, array):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    same for tf tensors and keras layers
    """
    # assert np.count_nonzero([array, tensor, layer]) <= 1, \
    #     'Only 1 of [array (={}), tensor (={}), layer (={})] can be defined!'.format(array, tensor, layer)

    text = text.ljust(25)
    if type(array) == np.ndarray:
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:8.2f}  mean: {:8.2f}  max: {:8.2f}".format(array.min(), array.mean(), array.max()))
        else:
            text += ("min: {:10}  mean:{:10}  max: {:10}".format("", "", ""))
        text += "  dtype: {}".format(array.dtype).rjust(15)
    elif type(array) == list or type(array) == tuple:
        text += ("length: {:20}  ".format(str(len(array))))
        #text += ("min: {:8.2f}  mean: {:8.2f}  max: {:8.2f}".format(min(array), mean(array), max(array)))
        text += "  type: {}".format(type(array[0])).rjust(15)
    elif type(array) == dict:
        for key, val in array.items():
            text += '{}: {}'.format(key, val)
    else:
        text += str(array)

    print(text)

def logVerbose(bLogArray=True, bLogTensor=True, bLogLayer=True):
    '''
    wrapper around the log function
    :param bLogArray:
    :param bLogTensor:
    :param bLogKeras:
    :return:
    '''

    # make a dummy function that
    # absorbs all arguments and returns tensors, also keras-layers needed?
    def f(*args, **kwargs):
        if 'tensor' in kwargs:
            if bLogTensor:
                return log(*args, tensor=kwargs['tensor'])
            else:
                return kwargs['tensor']
        elif 'layer' in kwargs:
            if bLogLayer:
                return log(*args, layer=kwargs['layer'])
            else:
                return kwargs['layer']
        elif 'array' in kwargs:
            if bLogArray:
                return log(*args, array=kwargs['array'])
            else:
                return
    return f

def getTifFiles(projectFolder):
    tifs = [os.path.join(projectFolder, f) for f in os.listdir(projectFolder)
            if f.endswith('.tif') and not f.endswith('overview.tif')]
    return tifs

def sortLabelFiles(file):
    datestr = file.split('__')[-1]
    try:
        return datetime.strptime(datestr, "%D_%m_%Y")
    except ValueError:
        # the format doesn't match
        return 0

def getDateStr():
    return datetime.now().strftime('%d_%m_%Y')

def splitAll(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])

    return allparts

def bbox2poly(points):
    (t, l), (b, r) = points
    return [(t, l), (t, r), (b, r), (b, l)]

def poly2bbox(points):
    [(t, l), (t, r), (b, r), (b, l)] = points
    return [(t, l), (b, r)]

def importLabelMe(labeldict, type='bbox'):
    # type = bbox or point
    bboxes = []
    names = []
    for shape in labeldict['shapes']:
        label = shape['label']
        coor = shape['points']
        coor = np.asarray(coor)
        if type == 'bbox' and len(coor) != 2:
            continue
        elif type == 'point' and len(coor) != 1:
            continue
        bboxes.append(coor)
        names.append(label)

    return bboxes, names

def labelBoxToYOLOBox(labelbox, imgsize):
    '''
    converts a labelme bbox to YOLO bbox
    LabelMe box:
        - (top,left) (bottom, right) aka (x1, y1), (x2, y2)
        - absolute coordinates [0, imgsize]
    YOLO:
        - (center_x, center_Y), (box_width, box_height) aka (xc, yc), (w, h)
        - relative values [0, 1]
    '''
    (x1, y1), (x2, y2) = labelbox
    w, h = imgsize[:2]
    xc = (x1 + x2)/2/w
    yc = (y1 + y2)/2/h
    bw = abs(x1 - x2)/w
    bh = abs(y1 - y2)/h
    return float(xc), float(yc), float(bw), float(bh)

def labelmeToYOLO(bboxes, names, namedict, imgsize):
    '''
    '''
    w, h = imgsize[:2]
    yboxes = [ labelBoxToYOLOBox(box, imgsize) for box in bboxes]
    ynames = [namedict[n] for n in names]

    return yboxes, ynames





