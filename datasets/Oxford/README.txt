
			    OXFORD and EF DATASET


This dataset  is the  "Oxford" and  "EF" dataset.  The  original images  for the
"Oxford" dataset  is from [1]  and for  the "EF" dataset  they are from  [2]. We
claim no credit for these datasets and we are providing them here simply for the
ease of using them with our software [3].

When using  this dataset, please refer  to the original distribution  of the two
datasets for their terms of use.

[1] K.  Mikolajczyk,  T.  Tuytelaars,  C.   Schmid, A.   Zisserman, J.   Matas,
    F. Schaffalitzky, T.  Kadir and L. Van Gool. "A  comparison of affine region
    detectors." International  journal of  computer vision  65, no.  1-2 (2005):
    43-72.

[2] L. Zitnick  and K. Ramnath. "Edge foci interest  points." In Computer Vision
    (ICCV), 2011 IEEE International Conference on, pp. 359-366. IEEE, 2011.

[3] Y.  Verdie, K.   M.  Yi,  P.  Fua,  and V.   Lepetit.  "TILDE:  A Temporally
    Invariant Learned DEtector.", Computer Vision and Patern Recognition (CVPR),
    2015 IEEE Conference on.

Contact:

Yannick Verdie : yannick<dot>verdie<at>epfl<dot>ch
Kwang Moo Yi   : kwang<dot>yi<at>epfl<dot>ch

================================================================================

Directory Structure

<Sequence Name>
	  |------ <test>
		    |------ <image_color>
		    |------ <image_gray>
		    |------ <homography>
		    |------ test_imgs.txt
		    |------ homography.txt

--
<test>  : Directory containing test and validation images
	<image_color>     : color images
	<image_gray>  	  : gray scale images (same naming with color images)
	<homography>  	  : homography relationship between images (3x3 matrices)
	
 	test_imgs.txt       : list of  image pairs to  be used  for validation
		   	      (image 1 is in pair with image N/2+1, where N is
		              the number of images)


