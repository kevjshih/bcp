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

result_str = fullfile(base_dir, sprintf('boost_model.mat'));
load(result_str, 'model');

%%%%%%%%%%%%%%%%%%%% Do It! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
calib_params = train_detector_calibration(D, cached_scores, cls);
[model.part.calib] = deal(calib_params{:});

[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');
cached_pos = cached_scores(pos_inds);

model_wbox = train_bbox_new(Dpos, cached_pos, model, calib_params);

cached_scores_box = box_inference(cached_scores, model_wbox);
model_wbox.learner = boost_iterate(D, cached_scores_box, cls, 5, 'sigmoid_java');

cached_scores_test_box = box_inference(cached_scores_test, model_wbox);

fprintf('Trained without box reprediction\n');
cached_scores_test_box = apply_weak_learner(cached_scores_test_box, model.learner);
roc50_a = test_given_cache(Dtest, cached_scores_test_box, cls, 0.5);
roc10_a = test_given_cache(Dtest, cached_scores_test_box, cls, 0.1, 1);

fprintf('Trained with box reprediction\n');
cached_scores_test_box = apply_weak_learner(cached_scores_test_box, model_wbox.learner);
roc50_b = test_given_cache(Dtest, cached_scores_test_box, cls, 0.5);
roc50_b = test_given_cache(Dtest, cached_scores_test_box, cls, 0.1, 1);

result_str = fullfile(base_dir, sprintf('loc_model.mat'));
save(result_str, 'model_wbox', 'roc*');


