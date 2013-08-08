function model = train_frombox_elda(I, bbox, sz, class, suffix)
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
data_dir = fullfile(sprintf('data/tmp/candidates_%s/', suffix), class);
outfile = fullfile(data_dir, data_file);

if(~exist(data_dir, 'file'))
   mkdir(data_dir);
end

if(~exist(outfile, 'file'))
   im = imread(I);

   f = warpfeat2(im, bbox, sz);
   model.x = f;
   model.hg_size = sz;

   model = train_whog_exemplar(model);
   
   models_name = sprintf('exemplar-elda-%s-%s', bn, mat2str(bbox));
   model.name = models_name;
   model.b= 0;
   model = convert_part_model(model);

   save(outfile, 'model');
end


if(~exist('model', 'var') && nargout>0)
    fprintf('Not recomputing\n');
   load(outfile, 'model');
end
