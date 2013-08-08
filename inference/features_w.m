% Usage for features mex function:
%   Unweighted features (same behavior as Felz. et al. features)
%   f = features_w(image, nbin)
%
%   Weighted features
%   f = features_w(image, weight_map, nbin)
%
%   image - Y x X x 3 (double); color image
%   weight_map - Y x X x 1 (double); weighting for each the gradient computed at each pixel.  Each weight can be any real value
%   nbin - number of pixels for each grid cell
