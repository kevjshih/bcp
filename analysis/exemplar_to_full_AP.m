javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

%cls = 'aeroplane';

if(1)
load_init_data;
end

empty_model = model;

%%%%%%%%%%%%%% Load models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

candidate_models = load_candidate_models(cls); % Use auto selected ones for now
Npart_candidates = length(candidate_models);
candidate_models = [candidate_models, load_candidate_models(cls, 0, 1)]; % Add object level 
Nobj_candidates = length(candidate_models) - Npart_candidates;

if(isempty(candidate_models))
   % Haven't trained them yet!!
   train_candidate_parts(cls, 1000); 
   candidate_models = load_candidate_models(cls);
%   cluster_test_candidates(cls, 10);
end

%%%%%%%%%%%%%% Create initial model %%%%%%%%%%%%%%%%%%%%%%%%%%

%model = train_region_model(D, cached_scores, model)
if(0)
   [labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

   cached_scores = apply_weak_learner(cached_scores, new_learner);
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);
end

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
usegt = 1; % Train on ground truth boxes only

if(usegt)
   cached_scores_gt = get_gt_pos_reg(D, cached_scores, cls); 
end

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
   elseif(usegt)
      if(~isempty(model.part(i).spat_const)) % This implies using object level template
         model.part(i).spat_const = [0 1 0 1 0.700 1]; % Use criteria of fgmr
      else
         model.part(i).spat_const = [0 1 0.9 1 0 1]; % Make sure part is mostly inside region
      end
      [model w_loo w_noloo] = train_loo_cache(model, D, cached_scores_gt, 10, 5, 1, 5.0);%, neg_feat_cache);
      w_all = w_loo;
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
   [labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls); % First using overlap criteria
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', [1:model.num_parts]);

   cached_scores = apply_weak_learner(cached_scores, new_learner);
   cached_scores = apply_weak_learner(cached_scores, new_learner);

   roc_50_overlap{i} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test_overlap{i} = test_given_cache(Dtest, cached_scores_test, cls, 0.5);

   % Retrain model with latent 
   [labels_sub cached_sub] = prune_boost_data([], cached_scores, []); % Train again with latent object position
%   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java', [1:model.num_parts]);
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
