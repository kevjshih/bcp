function [annotation, img, crop] = LMimresizecrop(annotation, img, M);
%
% [annotation, img, crop] = LMimresizecrop(annotation, img, M);;
%
% Output a square image of size M x M.


scaling = M/min([size(img,1) size(img,2)]);
[annotation, img] = LMimscale(annotation, img, scaling, 'bilinear');


[nr nc cc] = size(img);
sr = floor((nr-M)/2);
sc = floor((nc-M)/2);

[annotation, img, crop] = LMimcrop(annotation, img, [sc+1 sc+M sr+1 sr+M]);
