% Start by training dog model with additional cat parts!

cls = 'dog';

load_init_data;

load data/results/dog/incremental_model_transform_1.mat;

t = load('data/results/cat/incremental_model_maxp_5.mat', 'cached_scores', 'cached_scores_test', 'model');
%t = load('data/results/person/incremental_model_transform_2.mat', 'cached_scores', 'cached_scores_test');

cached_scores = merge_results(cached_scores, t.cached_scores);
cached_scores_test = merge_results(cached_scores_test, t.cached_scores_test);

clear t;

 [labels_sub cached_sub] = prune_boost_data_overlap_mc(D, cached_scores, {'dog'});

% Old stuff.....

%[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

cached_scores = apply_weak_learner(cached_scores, new_learner);
cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

roc_50{1} = test_given_cache(D, cached_scores, cls, 0.5);
roc_50_test{1} = test_given_cache(Dtest, cached_scores_test, cls, 0.5);

% Run again....
   [labels_sub cached_sub] = prune_boost_data(D, cached_scores, cls);
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

   cached_scores = apply_weak_learner(cached_scores, new_learner);
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

   roc_50{i+1} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test{i+1} = test_given_cache(Dtest, cached_scores_test, cls, 0.5,0,0);
