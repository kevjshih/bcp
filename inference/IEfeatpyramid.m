function [feat, scale] = featpyramid(im, sbin, interval)

% [feat, scale] = featpyramid(im, sbin, interval);
% Compute feature pyramid.
%
% sbin is the size of a HOG cell - it should be even.
% interval is the number of scales in an octave of the pyramid.
% feat{i} is the i-th level of the feature pyramid.
% scale{i} is the scaling factor used for the i-th level.
% feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

sc = 2 ^(1/interval);
imsize = [size(im, 1) size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));
feat = cell(max_scale, 1);
scale = zeros(max_scale, 1);

% our resize function wants floating point values
im = double(im);
for i = 1:interval
  scaled = IEresize(im, 1/sc^(i-1));
  % "second" 2x interval
  feat{i} = features_w(scaled, sbin);
  scale(i) = 1/sc^(i-1);
  % remaining interals
  for j = i+interval:interval:max_scale
    scaled = IEresize(scaled, 0.5);
    feat{j} = features_w(scaled, sbin);
    scale(j) = 1/sc^(j-1);
    %scale(j) = 0.5 * scale(j);
  end
end
