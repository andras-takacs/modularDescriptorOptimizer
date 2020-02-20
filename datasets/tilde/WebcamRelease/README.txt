
				WEBCAM DATASET


This dataset is the "Webcam" dataset, which appeared in [1]. We collected series
of images from outdoor webcams captured  at different times of day and different
seasons.   We identified  several  suitable  webcams from  the  Archive of  Many
Outdoor  Scenes (AMOS)  dataset [2]  -- webcams  that remained  fixed over  long
periods of  time, protected from the  rain, etc.  We also  used panoramic images
captured by a camera located on the top of a building. Source for each sequences
are as follows:

Chamonix   : AMOS dataset [2]
Courbevoie : AMOS dataset [2]
Frankfurt  : AMOS dataset [2]
Mexico	   : AMOS dataset [2]
StLouis	   : AMOS dataset [2]

Panorama   : Created at EPFL [1]

This dataset is strictly for academic purposes  only. For other purposes, please
contact us. When using this dataset, please cite [1] and [2].

[1] Y.  Verdie, K.   M.  Yi,  P.  Fua,  and V.   Lepetit.  "TILDE:  A Temporally
    Invariant Learned DEtector.", Computer Vision and Patern Recognition (CVPR),
    2015 IEEE Conference on.

[2] N. Jacobs, N. Roman, and  R. Pless.  "Consistent Temporal Variations in Many
    Outdoor Scenes."  Computer  Vision and Patern Recognition  (CVPR), 2007 IEEE
    Conference on.

Contact:

Yannick Verdie : yannick<dot>verdie<at>epfl<dot>ch
Kwang Moo Yi   : kwang<dot>yi<at>epfl<dot>ch

================================================================================

Directory Structure

<Sequence Name>
	  |------ <train>
	  |------ <test>
		    |------ <image_color>
		    |------ <image_gray>
		    |------ test_imgs.txt
		    |------ validation_imgs.txt

--
<train> : Directory containing training images and good keypoints to learn

	<file_name>.png	  : Image
	<file_name>_P.mat : Positive keypoints
	<file_name>_N.mat : Keypoints used as negatives

--
<test>  : Directory containing test and validation images
	<image_color>     : color images
	<image_gray>  	  : gray scale images (same naming with color images)
	
	validation_imgs.txt : list of  image pairs to  be used  for validation
		              (image 1 is in pair with image 11)
 	test_imgs.txt       : list of  image pairs to  be used  for validation
		              (image 1 is in pair with image 11)

--
Structure of the .mat files

[7 x N] matrices for N keypoints where each dimension is as follows:

1: x		: x coordinates
2: y		: y coordinates
3: <Not Used>	: Not used
4: repeatabiliy	: repeatability of the location
5: response	: SIFT scores from the location
6: scale	: SIFT scale of the location
7: <Not used>	: Not used


