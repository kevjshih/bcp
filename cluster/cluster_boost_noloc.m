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
   load_init_data;
end


base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
det_str = fullfile(base_dir, sprintf('part_detections.mat'));
load(det_str, 'model', 'cached_scores', 'cached_scores_test');

[cached_scores cached_scores_test] = estimate_missing_cached_data(cached_scores, cached_scores_test);

todo = [[1:model.num_parts],  model.num_parts + size(cached_scores{1}.region_score, 2) + 1];

result_str = fullfile(base_dir, sprintf('boost_noloc_model.mat'));

new_learner = boost_iterate(D, cached_scores, cls, 5, 'sigmoid_java', todo);

model.learner = new_learner;
cached_scores_test = apply_weak_learner(cached_scores_test, model.learner);

if(trainval==0)
   test_given_cache(Dtest, cached_scores_test, cls, 0.5);
   test_given_cache(Dtest, cached_scores_test, cls, 0.1, 1);
end

save(result_str, 'model');
