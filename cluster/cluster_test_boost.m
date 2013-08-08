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
   load_init_final;
   load_init_test;
else % Just the training set
   set_str = 'train';
   load_init_data;
end

base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
det_str = fullfile(base_dir, sprintf('part_detections.mat'));
load(det_str, 'cached_scores', 'cached_scores_test');


%%%%%%%%%%%%%%%%%%%% Do It! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Case 1: Just parts...\n');
result_str = fullfile(base_dir, sprintf('boost_noloc_model.mat'));
load(result_str, 'model');

[dk cached_scores_test_mean] = estimate_missing_cached_data(cached_scores, cached_scores_test);
clear dk 
cached_scores_test_mean = apply_weak_learner(cached_scores_test_mean, model.learner);

roc_00_50 = test_voc(Dtest, cached_scores_test_mean, cls, 0.5);
roc_00_10 = test_voc(Dtest, cached_scores_test_mean, cls, [0.1 0.5], 1);
clear cached_scores_test_mean;

fprintf('Case 2: Parts+Loc Feat\n');
result_str = fullfile(base_dir, sprintf('boost_model.mat'));
load(result_str, 'model');
cached_scores_test = apply_weak_learner(cached_scores_test, model.learner);

roc_10_50 = test_voc(Dtest, cached_scores_test, cls, 0.5);
roc_10_10 = test_voc(Dtest, cached_scores_test, cls, [0.1 0.5], 1);


fprintf('Case 3: Parts+Reloc\n');
result_str = fullfile(base_dir, sprintf('boost_noloc_model.mat'));
load(result_str, 'model');
result_str = fullfile(base_dir, sprintf('loc_model.mat'));
load(result_str, 'model_wbox');
cached_scores_test_box = box_inference(cached_scores_test, model_wbox);
[dk cached_scores_test_mean_box] = estimate_missing_cached_data(cached_scores, cached_scores_test_box);
clear dk;
cached_scores_test_mean_box = apply_weak_learner(cached_scores_test_mean_box, model.learner);

roc_01_50 = test_voc(Dtest, cached_scores_test_mean_box, cls, 0.5);
roc_01_10 = test_voc(Dtest, cached_scores_test_mean_box, cls, [0.1 0.5], 1);

clear cached_scores_test_mean_box;


fprintf('Case 4: Parts+reloc+shape feat\n');
result_str = fullfile(base_dir, sprintf('loc_model.mat'));
load(result_str, 'model_wbox');
cached_scores_test_box = apply_weak_learner(cached_scores_test_box, model_wbox.learner);

roc_11_50 = test_voc(Dtest, cached_scores_test_box, cls, 0.5);
roc_11_10 = test_voc(Dtest, cached_scores_test_box, cls, [0.1 0.5], 1);

result_str = fullfile(base_dir, sprintf('boost_eval.mat'));
save(result_str, 'roc_*');
