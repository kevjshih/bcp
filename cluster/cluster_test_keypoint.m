function cluster_test(cls, trainval, refinement_type)

% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
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


result_str = fullfile(base_dir, sprintf('kp_results.mat'));


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



for i = 1:model.num_parts
    if(~ismember(refinement_type, no_exemplar))
    [map(i) per_kp{i} keypoint_name{i} rocs(i)] = test_poselet_loc(model, Dpos_test, cached_gt_pos_test, i);
    end
   [map_med(i) per_kp_med{i} keypoint_name_med{i} rocs_med(i)] = test_poselet_loc(model_med, Dpos_test, cached_gt_pos_test, i);
end

%keyboard
save(result_str, 'model*', 'map*', 'per_kp*', 'keypoint_name*', 'rocs*');
