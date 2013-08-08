function [I, bbox] = auto_get_part(params, stream)
% [I, bbox] = auto_get_part(params, stream, amount, num_candidates)
%
% Automatically get a part, given a cell array of images/bboxes.
%
% Input:
%   params: parameters for the dataset, including paths to data
%   stream: cell array of structs containing the fields I, bbox, and id
%
% Output:
%   I: image path, or cell array of image paths
%   bbox: bounding box, or cell array of bounding boxes
%

I = {};
bbox = {};

for i = 1:length(stream)
   rec = stream{i};

   for j = 1:size(rec.bbox,1)
      I{end+1} = rec.I;
      bbox{end+1} = rec.bbox(j,:);
   end
end
