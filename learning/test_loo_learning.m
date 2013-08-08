[model w_loo w_noloo all_models] = train_loo(model, Dsub, cached_sub, 23, 5, 1, 1e-2); % Train on entire subset

w_all = repmat({w_noloo}, size(cached_scores));
w_all(sub_im_inds) = w_loo;

for i = 1:4
   cur_model = all_models{i*23};

%   [dk cached_scores_tmp] = collect_boost_data_loo(cur_model, D, cached_scores, w_all);
   [dk cached_scores_tmp] = collect_boost_data(cur_model, D, cached_scores);
   [dk cached_scores_test_tmp] = collect_boost_data(cur_model, Dtest, cached_scores_test);

   % Train model
   [labels_sub cached_sub] = prune_boost_data([], cached_scores_tmp, []); 
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

   % Apply model
   cached_scores_test_tmp = apply_weak_learner(cached_scores_test_tmp, new_learner);
   cached_scores_tmp = apply_weak_learner(cached_scores_tmp, new_learner);

   roc_compare{i} = test_given_cache(D, cached_scores_tmp, cls, 0.5);
   roc_compare_test{i} = test_given_cache(Dtest, cached_scores_test_tmp, cls, 0.5);
end

[dk cached_scores_tmp] = collect_boost_data_loo(model, D, cached_scores, w_all);
[dk cached_scores_test_tmp] = collect_boost_data(model, Dtest, cached_scores_test);

% Train model
[labels_sub cached_sub] = prune_boost_data([], cached_scores_tmp, []); 
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

% Apply model
cached_scores_test_tmp = apply_weak_learner(cached_scores_test_tmp, new_learner);
cached_scores_tmp = apply_weak_learner(cached_scores_tmp, new_learner);

roc_loo_spat = test_given_cache(D, cached_scores_tmp, cls, 0.5);
roc_loo_test_spat = test_given_cache(Dtest, cached_scores_test_tmp, cls, 0.5);

