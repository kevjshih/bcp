function m = train_exemplar(I, bbox, class)
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

BDglobals;
BDVOCinit;
VOCopts.sbin = 8;
VOCopts.trainset = 'train';
set_name = VOCopts.trainset;
params = VOCopts;
params.sbin = 8;
params.interval = 10;
params.MAXDIM = 10;

[dc bn dc] = fileparts(I);
I = sprintf(params.imgpath, bn);

data_file = sprintf('exemplar-elda-%s-%s-%s.mat', bn, mat2str(bbox), set_name);
data_dir = fullfile('data/tmp/candidates_exhaustive/', class);
outfile = fullfile(data_dir, data_file);

if(~exist(data_dir, 'file'))
   mkdir(data_dir);
end

if(~exist(outfile, 'file'))
   im = imread(I);
   models = initialize_exhaustive_model(im, bbox, params);
   fprintf('Training data_file %s: %d models!\n', data_file, length(models));


   parfor i = 1:length(models)
      fprintf('%d\n', i);
      m(i) = train_single_exemplar(params, I, bbox, class, set_name, models(i));
   end

   %randval = rand();  % Use this as a key to ensure consistency
   save(outfile, 'm', 'randval');
end



function m = train_single_exemplar(params, I, bbox, class, set_name, model)
[~, name, ~] = fileparts(I);

if(length(bbox)==4)
   MAXDIM = 10;
else
   MAXDIM = bbox(5);
   bbox = bbox(1:4);
end

bbox = round(bbox);

m.models_name = sprintf('exemplar-elda-%s-%s-%d-%s-%s', name, mat2str(bbox), MAXDIM, mat2str(round(model.bb)), set_name);
m.model = model;

%% Retrieve training set and truncate if necessary.
%train_set = get_pascal_set(params, set_name);

%% Add training set and training set's mining queue.
%m.train_set = train_set;

%% Get mining params.

% Trial 5-N [10 20 ...]
%% Add mining_params, and params to this exemplar.
%m.mining_params = [];
m.dataset_params = params;
%m.dataset_params.display = 0;
%% Go through mine-train iterations until enough images are mined.
%m.iteration = 1;

m.model = train_whog_exemplar(m.model);
m = convert_part_model(m);
