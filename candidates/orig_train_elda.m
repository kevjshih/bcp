function m = orig_train_elda(params, I, bbox, class, set_name, user_picked, object_part)
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
%   class: the class of object to train
%   set_name: PASCAL subset to mine negatives from
%   user_picked: whether the part was picked by a user or not
%
% Output:
%   m: the trained model, or cell array of trained models
%

if(~exist('object_part', 'var'))
   object_part = 0;
end

if(object_part==1)
   user_picked = 0;
end

if ~isa(I, 'cell')  % If I isn't a cell, then there must only be one.
    m = orig_train_single_exemplar(params, I, bbox, class, set_name, user_picked, object_part);
else  % Otherwise, we have to train many models.
    amount = numel(I);
    
    m = cell(1, amount);
   parfor i = 1:amount
        m{i} = orig_train_single_exemplar(params, I{i}, bbox{i}, class, set_name, user_picked, object_part);
    end
end
end

function m = orig_train_single_exemplar(params, I, bbox, class, set_name, user_picked, object_part)
[~, name, ~] = fileparts(I);


if(length(bbox)==4)
   MAXDIM = 10;
else
   MAXDIM = bbox(5);
   bbox = bbox(1:4);
end

bbox = round(bbox);

m.models_name = sprintf('exemplar-lda-%s-%s-%d-%s', name, mat2str(bbox), MAXDIM, set_name);

%% Create directory to cache the models in and load them if they already exist.
if user_picked
    basedir = sprintf('%s/user_exemplars/%s', params.localdir, class);
elseif object_part
    basedir = sprintf('%s/object_exemplars/%s', params.localdir, class);
else
    basedir = sprintf('%s/auto_exemplars/%s', params.localdir, class);
end
if ~exist(basedir,'dir')
    mkdir(basedir);
end

cached_filename = sprintf('%s/%s.mat', basedir, m.models_name);
if fileexists(cached_filename)
%   fprintf('File exists! Skipping\n');
%   load(cached_filename, 'm');
%   return;
   fprintf('File exists, but relearn anyways\n');
end

params.sbin = 8;
params.interval = 10;
params.MAXDIM = MAXDIM;

m.model = initialize_goalsize_model(convert_to_I(I), bbox, params);

if(isempty(m.model))
    return;
end

m.model = train_whog_exemplar(m.model);
%% Save model
save(cached_filename, 'm');
end
