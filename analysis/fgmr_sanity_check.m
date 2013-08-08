javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

%cls = 'aeroplane';

if(1)
load_init_fgmr_data;

for i = 1:length(cached_scores)
   cached_scores{i}.part_scores = zeros(numel(cached_scores{i}.labels),1);
   cached_scores{i}.part_boxes = zeros(numel(cached_scores{i}.labels),1);
end
for i = 1:length(cached_scores_test)
   cached_scores_test{i}.part_scores = zeros(numel(cached_scores_test{i}.labels),1);
   cached_scores_test{i}.part_boxes = zeros(numel(cached_scores_test{i}.labels),1);
end

end
empty_model = model;

%%%%%%%% Add root filters (v3, 3 components) %%%%%%%%
t = load(fullfile('data', 'fgmr_pretrained_v3', [cls '_hard.mat']));

fgmr_models = fgmr2boost(model, t.model);

model.score_feat = 0; % No incremental training yet

% Collect initial scores
[dk cached_scores_fgmr0] = collect_boost_data(fgmr_models, D, cached_scores);
[dk cached_scores_test_fgmr0] = collect_boost_data(fgmr_models, Dtest, cached_scores_test);

[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores_fgmr0, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);

cached_scores_fgmr0 = apply_weak_learner(cached_scores_fgmr0, new_learner);

[labels_sub cached_sub] = prune_boost_data([], cached_scores_fgmr0, []);  % Retraining again, selecting better regions
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);


%% Test it

cached_scores_fgmr0 = apply_weak_learner(cached_scores_fgmr0, new_learner);
cached_scores_test_fgmr0 = apply_weak_learner(cached_scores_test_fgmr0, new_learner);


roc_fgmr0 = test_given_cache(D, cached_scores_fgmr0, cls, 0.5);
roc_test_fgmr0 = test_given_cache(Dtest, cached_scores_test_fgmr0, cls, 0.5);


return; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5


% 1) Refine them using all regions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:fgmr_models.num_parts
   model.num_parts = i;
   if(i==1)
      model.part = fgmr_models.part(i);
   else
      model.part(i) = fgmr_models.part(i);
   end
   [model w_loo] = train_loo_cache(model, D, cached_scores, 10, 5, 1, 5.0);

   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_loo);
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);

   model.part(i).computed = 1;
end

[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);

cached_scores = apply_weak_learner(cached_scores, new_learner);

[labels_sub cached_sub] = prune_boost_data([], cached_scores, []);  % Retraining again, selecting better regions
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);


%% Test it

cached_scores = apply_weak_learner(cached_scores, new_learner);
cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);


roc = test_given_cache(D, cached_scores, cls, 0.5);
roc_test = test_given_cache(Dtest, cached_scores_test, cls, 0.5);

% 2) Refine them using best region %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cached_scores2 = cached_scores;
cached_scores_test2 = cached_scores_test;

for i = 1:length(cached_scores2)
   cached_scores2{i}.part_scores = zeros(numel(cached_scores2{i}.labels),0);
   cached_scores2{i}.part_boxes = zeros(numel(cached_scores2{i}.labels),0);
end
for i = 1:length(cached_scores_test2)
   cached_scores_test2{i}.part_scores = zeros(numel(cached_scores_test2{i}.labels),0);
   cached_scores_test2{i}.part_boxes = zeros(numel(cached_scores_test2{i}.labels),0);
end

model2 = model;
model2.num_parts = 0;
model2.part = [];

cached_sub = get_gt_pos_reg(D, cached_scores2, cls);
%cached_sub = get_best_pos_reg(D, cached_scores2, cls);

for i = 2:fgmr_models.num_parts
   model2.num_parts = i;
   if(i==1)
      model2.part = fgmr_models.part(i);
   else
      model2.part(i) = fgmr_models.part(i);
   end
   [model2 w_loo] = train_loo_cache(model2, D, cached_sub, 10, 5, 1, 5.0);

   % Compute Part Scores
   [labels cached_scores2] = collect_boost_data_loo(model2, D, cached_scores2, w_loo);
   [labels_test cached_scores_test2] = collect_boost_data(model2, Dtest, cached_scores_test2);

   model2.part(i).computed = 1;
end

[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores2, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);

cached_scores2 = apply_weak_learner(cached_scores2, new_learner);

[labels_sub cached_sub] = prune_boost_data([], cached_scores2, []);  % Retraining again, selecting better regions
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);


%% Test it

cached_scores2 = apply_weak_learner(cached_scores2, new_learner);
cached_scores_test2 = apply_weak_learner(cached_scores_test2, new_learner);


roc2 = test_given_cache(D, cached_scores2, cls, 0.5);
roc_test2 = test_given_cache(Dtest, cached_scores_test2, cls, 0.5);

% 2) Refine them using best region %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cached_scores2 = cached_scores;
cached_scores_test2 = cached_scores_test;

for i = 1:length(cached_scores2)
   cached_scores2{i}.part_scores = zeros(numel(cached_scores2{i}.labels),0);
   cached_scores2{i}.part_boxes = zeros(numel(cached_scores2{i}.labels),0);
end
for i = 1:length(cached_scores_test2)
   cached_scores_test2{i}.part_scores = zeros(numel(cached_scores_test2{i}.labels),0);
   cached_scores_test2{i}.part_boxes = zeros(numel(cached_scores_test2{i}.labels),0);
end

model2 = model;
model2.num_parts = 0;
model2.part = [];

cached_sub = get_best_pos_reg(D, cached_scores2, cls);

for i = 2:fgmr_models.num_parts
   model2.num_parts = i;
   if(i==1)
      model2.part = fgmr_models.part(i);
   else
      model2.part(i) = fgmr_models.part(i);
   end
   [model2 w_loo] = train_loo_cache(model2, D, cached_sub, 10, 5, 1, 5.0);

   % Compute Part Scores
   [labels cached_scores2] = collect_boost_data_loo(model2, D, cached_scores2, w_loo);
   [labels_test cached_scores_test2] = collect_boost_data(model2, Dtest, cached_scores_test2);

   model2.part(i).computed = 1;
end

[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores2, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);

cached_scores2 = apply_weak_learner(cached_scores2, new_learner);

[labels_sub cached_sub] = prune_boost_data([], cached_scores2, []);  % Retraining again, selecting better regions
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);


%% Test it

cached_scores2 = apply_weak_learner(cached_scores2, new_learner);
cached_scores_test2 = apply_weak_learner(cached_scores_test2, new_learner);


roc2 = test_given_cache(D, cached_scores2, cls, 0.5);
roc_test2 = test_given_cache(Dtest, cached_scores_test2, cls, 0.5);
% 3) Refine them using gt region %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cached_scores3 = cached_scores;
cached_scores_test3 = cached_scores_test;

for i = 1:length(cached_scores3)
   cached_scores3{i}.part_scores = zeros(numel(cached_scores3{i}.labels),0);
   cached_scores3{i}.part_boxes = zeros(numel(cached_scores3{i}.labels),0);
end
for i = 1:length(cached_scores_test3)
   cached_scores_test3{i}.part_scores = zeros(numel(cached_scores_test3{i}.labels),0);
   cached_scores_test3{i}.part_boxes = zeros(numel(cached_scores_test3{i}.labels),0);
end

model3 = model;
model3.num_parts = 0;
model3.part = [];

cached_sub = get_gt_pos_reg(D, cached_scores3, cls);
%cached_sub = get_best_pos_reg(D, cached_scores3, cls);

for i = 1:fgmr_models.num_parts
   model3.num_parts = i;
   if(i==1)
      model3.part = fgmr_models.part(i);
   else
      model3.part(i) = fgmr_models.part(i);
   end

   model3.part(i).spat_const = [0 1 0 1 0.700 1]; % Exact criteria of fgmr
   [model3 w_loo] = train_loo_cache(model3, D, cached_sub, 10, 5, 1, 5.0);

   % Compute Part Scores
   [labels cached_scores3] = collect_boost_data_loo(model3, D, cached_scores3, w_loo);
   [labels_test cached_scores_test3] = collect_boost_data(model3, Dtest, cached_scores_test3);

   model3.part(i).computed = 1;
end

[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores3, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);

cached_scores3 = apply_weak_learner(cached_scores3, new_learner);

[labels_sub cached_sub] = prune_boost_data([], cached_scores3, []);  % Retraining again, selecting better regions
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', 1:3);


%% Test it

cached_scores3 = apply_weak_learner(cached_scores3, new_learner);
cached_scores_test3 = apply_weak_learner(cached_scores_test3, new_learner);


roc3 = test_given_cache(D, cached_scores3, cls, 0.5);
roc_test3 = test_given_cache(Dtest, cached_scores_test3, cls, 0.5);

%%%%%%%%%%%%%% Create initial model %%%%%%%%%%%%%%%%%%%%%%%%%%

%model = train_region_model(D, cached_scores, model)
[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

cached_scores = apply_weak_learner(cached_scores, new_learner);
cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);


%%%%%%% Get filename (we want to avoid overwriting previous results) %%%%%%%%%
if(~exist(fullfile('data/results', cls), 'file'))
   mkdir(fullfile('data/results', cls));
end

fname0 = fullfile('data/results', cls, 'indep_exemplar_refinement_%d.mat');

ind = 1;
while(1)
   if(exist(sprintf(fname0, ind), 'file'))
      ind = ind + 1;
   else
      break;
   end
end

fname = sprintf(fname0, ind);
system(sprintf('touch %s', fname));

%%%%%%%%%%%%% Select parts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%candidate_models = convert_part_model(convert_part_model(candidate_models)); % hacky way to remove irrelevant space consuming fields

model.hard_local = 0;
model.score_feat = 0;

% Generate learning schedule.  This could be reordered based on current performance
% This should be precomputed and saved ...
[pos_prec chosen init_aps] = choose_candidate_amp(model, D, cached_scores, candidate_models);

exemplar_aps = init_aps(chosen);
subset = 0;
for i = 1:length(chosen)
%% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   best_model = candidate_models{chosen(i)};

%% Update model bookkeeping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   model = add_model(model, best_model); %, 1); % 1 indicates adding a spatial model
   chosen_names{i} = best_model.name;
   
%% Learn to use spatial features: %%%%%%%%%%%%%%%%%%%%%%%
   if(subset)
      [Dsub cached_sub sub_im_inds] = collect_pos_subset(cls, D, cached_scores, pos_prec{chosen(i)}, 0.3);
      [model w_loo w_noloo] = train_loo(model, Dsub, cached_sub, 10, 5, 1, 1e-2);%, neg_feat_cache);
      w_all = repmat({w_noloo}, size(cached_scores));
      w_all(sub_im_inds) = w_loo;
   else
      [model w_loo w_noloo] = train_loo(model, D, cached_scores, 10, 5, 1, 1e-2);%, neg_feat_cache);
%      w_all = repmat({w_noloo}, size(cached_scores));
%      w_all(sub_im_inds) = w_loo;
      w_all = w_loo;
   end
%% Boosting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
%   [labels cached_scores] = collect_boost_data(model, D, cached_scores);
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);
   model.part(i).computed = 1;

   % Train model
   [labels_sub cached_sub] = prune_boost_data([], cached_scores, []); 
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

   % Apply model
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);
   cached_scores = apply_weak_learner(cached_scores, new_learner);

   roc_50{i} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test{i} = test_given_cache(Dtest, cached_scores_test, cls, 0.5);

   [recall prec refined_aps(i)] = test_part_detections_D(cls, D, cached_scores, i);
   [recall prec refined_test_aps(i)] = test_part_detections_D(cls, Dtest, cached_scores_test, i);
 
   save(fname, 'model', 'chosen_names', 'roc_50_test', 'roc_50', 'cached_scores', 'cached_scores_test', 'exemplar_aps', 'refined_aps', 'refined_test_aps');
end
