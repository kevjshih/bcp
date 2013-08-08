javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

load_init_data;
empty_model = model;

%%%%%%%%%%%%%% Load models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
candidate_file = fullfile('data', [cls '_candidates.mat']);

if(~exist(candidate_file, 'file'))
   candidate_models = load_candidate_models(cls); % Use auto selected ones for now
   Npart_candidates = length(candidate_models);
   candidate_models = [candidate_models, load_candidate_models(cls, 0, 1)]; % Add object level 
   Nobj_candidates = length(candidate_models) - Npart_candidates;
   
   % Generate learning schedule.  This could be reordered based on current performance
   % This should be precomputed and saved ...
   [pos_prec chosen aps] = choose_candidate_amp(model, D, cached_scores, candidate_models);
   save(candidate_file, 'pos_prec', 'chosen', 'aps', 'candidate_models');
else
   load(candidate_file, 'pos_prec', 'chosen', 'aps', 'candidate_models');
end

if(isempty(candidate_models))
   % Haven't trained them yet!!
   train_candidate_parts(cls, 2000); 
   candidate_models = load_candidate_models(cls);
%   cluster_test_candidates(cls, 10);
end

%%%%%%%%%%%%%% Create initial model %%%%%%%%%%%%%%%%%%%%%%%%%%

%model = train_region_model(D, cached_scores, model)
[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

cached_scores = apply_weak_learner(cached_scores, new_learner);
cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

roc_50{1} = test_given_cache(D, cached_scores, cls, 0.5);
roc_50_test{1} = test_given_cache(Dtest, cached_scores_test, cls, 0.5);

%%%%%%% Get filename (we want to avoid overwriting previous results) %%%%%%%%%
if(~exist(fullfile('data/results', cls), 'file'))
   mkdir(fullfile('data/results', cls));
end

fname0 = fullfile('data/results', cls, 'incremental_model_subset_%d.mat');

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


% Generate Random learning schedule, with fixed proportion of expected samples from parts (75%) and objects (25%)
if(0)
   chosen = choose_candidate_random(candidate_models);
end

% Set up local flags
subset_predefined = 0;
subset_search = 0;
usegt = 0;
do_spatial_model = 0;


% Set up model flags
model.hard_local = 0;

% Incremental learning stuff
model.score_feat = 0;
model.incremental_feat = 0;
if(model.incremental_feat)
   model.num_incremental_feat = 5;
   model.cached_weight = zeros(1, model.num_incremental_feat);
end
% ....
model.do_transform = 1;
subset_split_size = 2; % Split examples into two partitions
model.shift = [0]; % No shift for now, but we want
model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but we want

if(usegt || do_spatial_model)
   cached_gt = get_gt_pos_reg(D, cached_scores, cls);
end


for i = 1:length(chosen)
   if(i<=model.num_parts)
      fprintf('Skipping part %d\n', i);
      continue;
   end
%% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   best_model = candidate_models{chosen(i)};

   model = add_model(model, best_model, do_spatial_model); %, 1); % 1 indicates adding a spatial model
   model.part(i).bias = model.part(i).bias + 0.5; % Make sure to pull in plenty of hard negatives in the first iteration
   chosen_names{i} = best_model.name;


%% Update model bookkeeping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   if(isfield(model.part(i), 'spat_const') && ~isempty(model.part(i).spat_const) && model.part(i).spat_const(5)>0.5) % Use FGMR criteria
      model.part(i).spat_const(5) = 0.5;
   else
      model.part(i).spat_const = [0 1 0.8 1 0 1];
   end

%% Learn to use spatial features: %%%%%%%%%%%%%%%%%%%%%%%
   if(usegt)
      cached_train = cached_gt;
   else
      cached_train = cached_scores;
   end

   % Begin by training on subset
   model.subset_split = 0;
   model = train_loo_cache(model, D, cached_train, 10, 2, 0.3, 5.0);%, neg_feat_cache);
   
   % Assign initial labels
   model.subset_split = subset_split_size;
   [model w_loo w_noloo all_models] = train_split_cache(model, D, cached_train, 10, 5, 0.3, 5.0);%, neg_feat_cache);
   w_all = w_loo;

%% Boosting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
   %[labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);
   model.part(i).computed = 1;

   % Train model
%   for boost_iter = 1:10
%      if(boost_iter == 1) % Start from scratch every time.
%         [labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
%      else % Search for highest scoring region
         [labels_sub cached_sub] = prune_boost_data([], cached_scores, []);
%      end

      new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');
      cached_scores = apply_weak_learner(cached_scores, new_learner);

%   end

   % Apply model to test
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

   roc_50{i+1} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test{i+1} = test_given_cache(Dtest, cached_scores_test, cls, 0.5,0,0);
 
   roc_10{i+1} = test_given_cache(D, cached_scores, cls, 0.1, 1);
   roc_10_test{i+1} = test_given_cache(Dtest, cached_scores_test, cls, 0.1, 1);

   save(fname, 'model', 'chosen_names', 'roc_50_test', 'roc_50', 'roc_10*','cached_scores', 'cached_scores_test');
end



return;
   for boost_iter = 4:10
      if(boost_iter == 1) % Start from scratch every time.
         [labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
      else % Search for highest scoring region
        [labels_sub cached_sub] = prune_boost_data([], cached_scores, []);
      end

     new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');
     %new_learner = boost_train(cached_sub, labels_sub, 'svm', 1:13, 0.1);
     cached_scores = apply_weak_learner(cached_scores, new_learner);
      cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

      roc_50_t{boost_iter} = test_given_cache(D, cached_scores, cls, 0.5);
      roc_50_test_t{boost_iter} = test_given_cache(Dtest, cached_scores_test, cls, 0.5);
      roc_50_test_voc{boost_iter} = test_voc(Dtest, cached_scores_test, cls, 0.5);

   end
