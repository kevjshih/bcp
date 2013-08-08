javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

cls = 'cat';

if(1)
load_init_data;
end

addpath(genpath('./'));
rmpath(genpath('./old'));
empty_model = model;

%%%%%%%%%%%%%% Load models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

candidate_models = load_candidate_models(cls, 1); % Use user selected ones for now

if(isempty(candidate_models))
   % Haven't trained them yet!!
   train_candidate_parts(cls, 1000); 
   candidate_models = load_candidate_models(cls);
%   cluster_test_candidates(cls, 10);
end

%%%%%%%%%%%%%% Add object level detectors %%%%%%%%%%%%%%%%%%%%%
if(0)
t = load(fullfile('data', [cls '_hard.mat']));

fgmr_models = fgmr2boost(model, t.model);

model.score_feat = 0; % No incremental training yet
% Refine them
for i = 2:fgmr_models.num_parts
   model.num_parts = i;
   if(i==1)
      model.part = fgmr_models.part(i);
   else
      model.part(i) = fgmr_models.part(i);
   end
   [model w_loo] = train_loo(model, D, cached_scores, 7, 7, 1);

   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_loo);
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);

   model.part(i).computed = 1;
end

 
[labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
[labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);

[model.part.computed] = deal(1);
end %if(0)
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

fname0 = fullfile('data/results', cls, 'incremental_model_maxp_%d.mat');

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

candidate_models = convert_part_model(convert_part_model(candidate_models)); % hacky way to remove irrelevant space consuming fields

model.hard_local = 0; % Search for hard negative localization errors?
model.score_feat = 1; % Include previous round's boosting score as a feature?

% Generate learning schedule.  This could be reordered based on current performance
% This should be precomputed and saved ...
[pos_prec chosen] = choose_candidate_amp(model, D, cached_scores, candidate_models);

new_incremental = 1;

for i = 1:length(chosen)
   % Select data subset, etc
   best_model = candidate_models{chosen(i)};

   if(new_incremental)
      [Dsub cached_sub] = collect_pos_subset(cls, D, cached_scores, pos_prec{chosen(i)}, 0.3);
   end

   % Update model bookkeeping
   model = add_model(model, best_model);
   chosen_names{i} = best_model.name;
   
   % Refine Model
   %[model w_loo] = train_loo(model, Dsub, cached_sub, 7, 7, 1, 1e-2); % Train on entire subset
   if(new_incremental)
      model = train_loo(model, Dsub, cached_sub, 7, 7, 1, 1e-2); % Train on entire subset
   else
      model = train_loo(model, D, cached_scores, 7, 7, .25, 1e-2); % Search for subset
   end
   % Compute Part Scores
   %[labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_loo); % TODO!
   [labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);

   % Train model
   [labels_sub cached_sub] = prune_boost_data([], cached_scores, []); 
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

   % Apply model
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);
   cached_scores = apply_weak_learner(cached_scores, new_learner);

   roc_50{i} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test{i} = test_given_cache(Dtest, cached_scores_test, cls, 0.5);
 
   model.part(i).computed = 1;
   save(fname, 'model', 'chosen_names', 'roc_50_test', 'roc_50', 'cached_scores', 'cached_scores_test');
end
