function [imgCrop, scaling] = LMobjectnormalizedcrop(img, annotation, j, b, height, width)
%
% Crop object from image using a normalized frame
% [imgCrop, scaling] = LMobjectnormalizedcrop(img, annotation, j, b, height, width)
%  extract object index j from the annotation.
%
%   ------------
%   |     b    |   b = number of boundary pixels
%   |   ----   |   h = height inner bounding box
%   | b |  |   |   w = width inner bounding box
%   |   |h |   |
%   |   ----   |
%   |    w     |
%   ------------
%

[nrows ncols cc] = size(img);

[x,y] = getLMpolygon(annotation.object(j).polygon);
bb = [min(x) min(y) max(x) max(y)];
%  boundingbox = [xmin ymin xmax ymax]

% 1) Resize image so that object is normalized in size
scaling = min(height/(bb(4)-bb(2)), width/(bb(3)-bb(1)));
[annotation, img] = LMimscale(annotation, img, scaling, 'bilinear');

% 2) pad image (just to make sure)
[annotation, img] = LMimpad(annotation, img, [size(img,1)+2*b+3 size(img,2)+2*b+3], 128);

% 2) Crop result
[x,y] = getLMpolygon(annotation.object(j).polygon);
cx = (max(x)+min(x))/2; cy = (max(y)+min(y))/2;
bb = [cx-width/2-b cy-height/2-b cx+width/2+b cy+height/2+b];

% Image crop:
imgCrop = img(bb(2):bb(4), bb(1):bb(3), :);

