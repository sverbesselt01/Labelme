# AI for orthomosaics

### Requirements
- Pycharm (community edition)
- Python 3.12
- QGis 3.40.4

### Install python modules
1. Open terminal
2. Type following command in the terminal
~~~shell
pip install -r requirements.txt
~~~

:rocket: Ready to start :rocket:

### Processing steps
0. Modify the hyperparameters in the .env file. 
1. drawRaster.py: makes raster of tiles for DL model, select buffer to be large enough to cover full orthomosaic.
2. correctRaster.py: remove empty boxes (with no pixels).
3. In QGIS: Make a copy of your tile file and select the tiles that have a dead tree within the tile (data of Thomas).
4. extractTiles.py: crop the orthomosaic to the extent of every box.
4. Label the images using labelme.
5. Convert_json_to_polygons.py: make shape file with labels.
6. Merge the alive trees and dead trees, dissolve boundaries. Make sure the dead trees are on top of the alive trees (shape_labels_merging.model3 script in QGIS). 
7. Scatterplot_and_correlation.py: Calculate the amount (in %) of dead trees, alive trees, soil for every tile do correlation between segemented trees (labelme) and result Thomas (NDVI thresholds + CHM threshold).

