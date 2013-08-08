function model = simple_train_elda(I, bbox)
% m = orig_train_exemplar(params, I, bbox, class, set_name)
%
% Train a model from a single bounding box in an image using the original
% exemplar svm code.
%
% Modified from voc_template.m and train_all_exemplars.m
%
% Input:
%   params: parameters for the dataset, including paths to data
%   I: path of image, or cell array of paths of images, to train on
%   bbox([x1 y1 x2 y2]): bounding box, or cell array of bounding boxes, in image of object
%       if bbox has length 5, bbox(5) corresponds to the max allowed size of the model in HOG cells.
%
% Output:
%   m: the trained model

[~, name, ~] = fileparts(I);


if(length(bbox)==4)
   MAXDIM = 10;
else
   MAXDIM = bbox(5);
   bbox = bbox(1:4);
end

bbox = round(bbox);

models_name = sprintf('exemplar-lda-%s-%s-%d', name, mat2str(bbox), MAXDIM);

params.sbin = 8;
params.interval = 10;
params.MAXDIM = MAXDIM;

model = initialize_goalsize_model(convert_to_I(I), bbox, params);
model = train_whog_exemplar(model);
model.name = models_name;

model = convert_part_model(model);
