The folders contain images, homographies and the edge foci interest points for the data used in the paper - Edge Foci Interest Points.

If you use this data, please reference the following paper:

Edge foci interest points
C. L. Zitnick, K. Ramnath
ICCV, 2011

The format of the interest point files is:

NumofPoints
x y strength sxx sxy syy
...


to get scale take sqrt of sxx or syy.

The homographies are in the same format (3x3 transformation matrix) as the data at:
http://www.robots.ox.ac.uk/~vgg/research/affine/