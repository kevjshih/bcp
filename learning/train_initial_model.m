function [m, tfiles, models_name] = train_model(im, bbox, cls)
% Input:
%   im - path of image to train on
%   bbox - bounding box in image of object
%   cls - class of object

BDglobals

%% Initialize model
m.model = initialize_goalsize_model(convert_to_I(im), bbox);
[~, name, ~] = fileparts(im);
m.models_name = sprintf('%s-%s', name, mat2str(bbox));
m.name = m.models_name;
m.cls = cls;
m.curid = '0';
m.objectid = 0;

%% Train model
dataset_params = get_voc_dataset(VOCYEAR, WORKDIR, PASCALDIR);
dataset_params.display = 0;
dataset_params.SKIP_EVAL = 0; %Do not skip evaluation, unless it is VOC2010

mining_params = get_default_mining_params;
mining_params.set_name = 'train';%['-' cls];
mining_params.training_function = @do_svm;
dataset_params.mining_params = mining_params;
dataset_params.params = mining_params;

curparams = dataset_params.mining_params;

% Train on everything but cls of box
train_set = get_pascal_set(dataset_params, dataset_params.mining_params.set_name, ['-' cls]);

if isfield(curparams,'set_maxk')
    train_set = train_set(1:min(length(train_set), ...
        curparams.set_maxk));
end

[tfiles, models_name] = train_all_exemplars(dataset_params, {m}, train_set);

end
