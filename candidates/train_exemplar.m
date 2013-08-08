function m = train_exemplar(params, I, bbox, class, set_name)
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

[dc bn dc] = fileparts(I);
I = sprintf(params.imgpath, bn);

if ~isa(bbox, 'cell')  % If I isn't a cell, then there must only be one.
    m = orig_train_single_exemplar(params, I, bbox, class, set_name);
else  % Otherwise, we have to train many models.
    amount = numel(bbox);
    
    m = cell(1, amount);
   %par
   parfor i = 1:amount
        m{i} = orig_train_single_exemplar(params, I, bbox{i}, class, set_name);
    end
end
end

function m = orig_train_single_exemplar(params, I, bbox, class, set_name)
[~, name, ~] = fileparts(I);


if(length(bbox)==4)
   MAXDIM = 10;
else
   MAXDIM = bbox(5);
   bbox = bbox(1:4);
end

bbox = round(bbox);

m.models_name = sprintf('exemplar-svm-%s-%s-%d-%s', name, mat2str(bbox), MAXDIM, set_name);


%% Get mining params.
mining_params = get_default_mining_params();
mining_params.training_function = @do_svm;

% Trial 5-N [10 20 ...]
mining_params.MAX_TOTAL_MINED_IMAGES = 300; % Do a smaller subset with lots of svm updates
mining_params.MAX_IMAGES_BEFORE_SVM = 20;
mining_params.MAX_MINE_ITERATIONS = 1000;

params.sbin = 8;
params.interval = 10;
params.MAXDIM = MAXDIM;

m.model = initialize_goalsize_model(convert_to_I(I), bbox, params);

%% Retrieve training set and truncate if necessary.
train_set = get_pascal_set(params, set_name);

if isfield(params,'set_maxk')
    train_set = train_set(1:min(length(train_set), params.set_maxk));
end

%% Add training set and training set's mining queue.
m.train_set = train_set;
m.mining_queue = initialize_mining_queue(m.train_set);

%% Add mining_params, and params to this exemplar.
m.mining_params = mining_params;
m.dataset_params = params;
m.dataset_params.display = 0;
%% Go through mine-train iterations until enough images are mined.
m.iteration = 1;

% The mining queue is the ordering in which we process new images
while true
    if ~isfield(m,'mining_stats')
        total_mines = 0;
    else
        total_mines = sum(cellfun(@(x)x.total_mines,m.mining_stats));
    end
    m.total_mines = total_mines;
    m = mine_train_iteration(m, mining_params.training_function);
    
    if ((total_mines >= mining_params.MAX_TOTAL_MINED_IMAGES) || ...
            (isempty(m.mining_queue))) || ...
            (m.iteration == mining_params.MAX_MINE_ITERATIONS)
        fprintf(1,'Mined enough images, rest up\n');
        fprintf(1,' ##Breaking because we reached end\n');
        break;
    end
    
    m.iteration = m.iteration + 1;
end

end
