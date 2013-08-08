javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

cls = 'cat';

if(1)
load_init_data;
end
empty_model = model;

%%%%%%%%%%%%%% Load models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

candidate_models = load_candidate_models(cls);

if(isempty(candidate_models))
   % Haven't trained them yet!!
   train_candidate_parts(cls, 1000); 
   candidate_models = load_candidate_models(cls);
   cluster_test_candidates(cls, 10);
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

fname0 = fullfile('data/results', cls, 'incremental_model_%d.mat');

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

remaining_models = convert_part_model(convert_part_model(candidate_models));

model.hard_local = 0;
model.score_feat = 1;

for i = 1:1000%length(candidate_models)
   if(1)
      [best_model best_ind] = select_candidate_parts(model, D, cached_scores, remaining_models, 1); 
   elseif(0)
      best_ind = ceil(rand*numel(remaining_models));
      best_model = remaining_models{best_ind};
   else
      best_ind = i*20;
      best_model = remaining_models{best_ind};
   end

   % Update model bookkeeping
   model = add_model(model, best_model);
   chosen_names{i} = best_model.name;
   %remaining_models(best_ind) = [];
   
   % Refine Model
%   [model] = train_loo(model, D, cached_scores, 7, 7, 1); 
   [model w_loo] = train_loo(model, D, cached_scores, 7, 7, 0.15, 1e-1);
%   [model w_loo] = train_weighted_loo(model, D, cached_scores, 7, 7, 1);

   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_loo);
   %[labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);

   % Train model
   %[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
   %new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');
   %cached_scores = apply_weak_learner(cached_scores, new_learner);

   [labels_sub cached_sub] = prune_boost_data([], cached_scores, []); 
%   [labels_sub cached_sub] = prune_boost_data([], cached_scores, i,1); 
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

   % Apply model
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);
   cached_scores = apply_weak_learner(cached_scores, new_learner);

   roc_50{i} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test{i} = test_given_cache(Dtest, cached_scores_test, cls, 0.5);
 
   model.part(i).computed = 1;
   save(fname, 'model', 'chosen_names', 'roc_50_test', 'roc_50', 'remaining_models', 'cached_scores', 'cached_scores_test');
end
