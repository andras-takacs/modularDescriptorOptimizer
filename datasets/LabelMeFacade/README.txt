03-Sep-2010, 

bjoern.froehlich@uni-jena.de

This is the LabelMeFacade Image Database Version 1.0

To extract a new database use the script in "labelme-perl".

This database contains 100 images for training and 845 images for testing (see train.txt and test.txt for details)

color codes for labels are (in RGB):
various = 0:0:0
building = 128:0:0
car = 128:0:128
door = 128:128:0
pavement = 128:128:128
road = 128:64:0
sky = 0:128:128
vegetation = 0:128:0
window = 0:0:128

If you use this database please cite the following paper:

@INPROCEEDINGS{Froehlich-Rodner-Denzler-ICPR2010,
	author = {Bj{\"o}rn Fr{\"o}hlich and Erik Rodner and Joachim Denzler},
	title = {A Fast Approach for Pixelwise Labeling of Facade Images},
	booktitle = {Proceedings of the International Conference on Pattern Recognition
	(ICPR 2010)},
	year = {2010},
}