function cluster_test(cls, trainval, refinement_type)

% Parse input
try
if(~isinf(str2num(cls)))
   cls = str2num(cls);
end
end

try
   trainval = str2num(trainval);
end

%matlabpool_robust;

%%%%%%%%%%%%%%%%% Setup class... %%%%%%%%%%%%%%%%%%%%%%%%%%
startup_cluster
BDglobals;

VOCinit;
VOCopts.sbin = 8;
VOCopts.localdir = fullfile(WORKDIR, 'exemplars');

if(isnumeric(cls))
   clsind = cls;
   cls = VOCopts.classes{clsind};
end

fprintf('Doing category: %s\n', cls);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(trainval)
   set_str = 'trainval';
   load_init_final;
   load_init_test;
else % Just the training set
   set_str = 'train';
   C  = 15;
   load_init_data;
end


base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
det_str = fullfile(base_dir, sprintf('part_detections.mat'));
load(det_str, 'model', 'cached_gt_pos_test', 'cached_gt_pos');


%result_str = fullfile(base_dir, sprintf('ss1_kp_results.mat'));


if(strfind(refinement_type, 'nosimilar'))
   similar = get_similar_images(D, cls);

   D(similar) = [];
end

Dpos = LMquery(D, 'object.name', cls, 'exact');
Dpos_test = LMquery(Dtest, 'object.name', cls, 'exact');

% Train poselet model... (just copy over keypoint annotations)
model_med = model;

no_exemplar = {'dpm', 'poselet'}; % Poselets are just to save time

for i = 1:model.num_parts
    if(~ismember(refinement_type, no_exemplar))
   model = train_exemplar_poselet(model, i);
    end
%   model_med = train_poselet_loc(model_med, Dpos, cached_gt_pos, i); % This version learns kp locations from the medians of 15 examples
   model_med = train_poselet_loc(model_med, Dpos, cached_gt_pos, i); % This version learns kp locations from the medians of 15 examples
end
% load data...
%load(['/projects/VisionLanguage/ss1/forIan/' cls '.mat']); % loads valBboxFeats, trainBboxFeats
%load('/projects/VisionLanguage/ss1/forIan/IMAGE_IDS.mat');

load(['data/ss1/' cls '.mat']); % loads valBboxFeats, trainBboxFeats
load('data/ss1/IMAGE_IDS.mat');

for i = 1:length(trainBboxFeats)
   [train_kp(i).ok train_kp(i).id] = test_poselet_saurabh(cls, model, trainIds{i}, trainBboxFeats(i));
end

for i = 1:length(valBboxFeats)
   [val_kp(i).ok val_kp(i).id] = test_poselet_saurabh(cls, model, valIds{i}, valBboxFeats(i));
end

result_str = fullfile(['data/ss1/' cls '_kp_results.mat']);

save(result_str, 'val_kp', 'train_kp');
