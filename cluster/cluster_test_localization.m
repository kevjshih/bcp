function cluster_localization(cls, trainval, refinement_type)

% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
end

try
   trainval = str2num(trainval);
end

%matlabpool_robust;

%%%%%%%%%%%%%%%%% Setup class... %%%%%%%%%%%%%%%%%%%%%%%%%%
startup_cluster;
get_cluster_basedir;
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
%   load_init_final; no need for this....
   load_init_test;
else % Just the training set
   set_str = 'train';
   load_init_data;
end


base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
loc_str = fullfile(base_dir, sprintf('loc_results.mat'));

if(~exist(loc_str, 'file'))

   det_str = fullfile(base_dir, sprintf('part_detections.mat'));
   load(det_str, 'cached_scores_test');

   model_str = fullfile(base_dir, sprintf('loc_model.mat'));
   load(model_str, 'model_wbox');

   %%%%%%%%%%%%%%%%%%%% Do It! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   cached_scores_test_box = box_inference(cached_scores_test, model_wbox);
   cached_scores_test_box = apply_weak_learner(cached_scores_test_box, model_wbox.learner);

   %roc50_b = test_given_cache(Dtest, cached_scores_test_box, cls, 0.5);
   save(loc_str, '-v7.3', 'cached_scores_test_box');
else
   load(loc_str, 'cached_scores_test_box');
end

eval_results(Dtest, cached_scores_test_box, cls, 'VOC2010', 'test');







