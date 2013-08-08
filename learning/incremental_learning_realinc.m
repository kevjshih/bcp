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

fname0 = fullfile('data/results', cls, 'incremental_model_realinc_%d.mat');

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
%subset_predefined = 0;
%subset_search = 0;
%usegt = 1; % This flag isn't used anymore, everything is trained with GT
subsetonly = 0;
subsetinit = 1;
do_spatial_model = 0;
incremental_selection = 0;
if(incremental_selection)
   chosen = [];
   not_chosen = ones(1, length(candidate_models));
end


% Set up model flags
model.hard_local = 0;
model.score_feat = 1;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but we want

cached_gt = get_gt_pos_reg(D, cached_scores, cls);

for i = 1:length(candidate_models)
   if(i<=model.num_parts)
      fprintf('Skipping part %d\n', i);
      continue;
   end
%% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   best_model = candidate_models{chosen(i)};

   model = add_model(model, best_model); %, 1); % 1 indicates adding a spatial model
   model.part(i).bias = model.part(i).bias + 0.5; % Make sure to pull in plenty of hard negatives in the first iteration
   chosen_names{i} = best_model.name;

%% Update model bookkeeping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   if(isfield(model.part(i), 'spat_const') && ~isempty(model.part(i).spat_const) && model.part(i).spat_const(5)>0.5) % Use FGMR criteria
      model.part(i).spat_const(5) = 0.5;
   else
   %   model.part(i).spat_const = [0 1 0.8 1 0 1];
   end


%% Learn to use spatial features: %%%%%%%%%%%%%%%%%%%%%%%
%   if(usegt) Always use gt!!!!
   if(subsetonly)
      [model neg_feats w_all w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 5, 0.5, 15.0);%, neg_feat_cache);
   elseif(subsetinit)
      model.score_feat = 0;
      model.cached_weight = 0; % Just to be safe...
      model = train_loo_cache(model, D, cached_gt, 10, 2, 0.5, 15.0); % Strong initialization with subset of the data
      
      model.score_feat = 1;
      [model neg_feats w_all w_noloo all_models] = train_loo_cache(model, D, cached_scores, 10, 3, 1, 15.0); % Refine with the rest to pick up any other examples
   else 
      [model neg_feats w_all w_noloo all_models] = train_loo_cache(model, D, cached_scores, 10, 5, 1, 15.0);%, neg_feat_cache);
   end
   
   clear neg_feats
%% Boosting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
   %[labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);
   model.part(i).computed = 1;

   [labels_sub cached_sub] = prune_boost_data(D, cached_scores, cls);
   %[labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

% Test it!
   cached_scores = apply_weak_learner(cached_scores, new_learner);
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

   roc_50{i+1} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test{i+1} = test_given_cache(Dtest, cached_scores_test, cls, 0.5,0,0);
 
   roc_10{i+1} = test_given_cache(D, cached_scores, cls, 0.1, 1);
   roc_10_test{i+1} = test_given_cache(Dtest, cached_scores_test, cls, 0.1, 1);

   save(fname, '-v7.3', 'model', 'chosen_names', 'roc_50_test', 'roc_50', 'roc_10*','cached_scores', 'cached_scores_test');
end



return;
   for boost_iter = 1:4
      if(boost_iter == 1) % Start from scratch every time.
         [labels_sub cached_sub imind] = prune_boost_data_overlap(D, cached_scores, cls);
      else % Search for highest scoring region
        [labels_sub0 cached_sub0 imind0] = prune_boost_data([], cached_scores, []);
        [labels_sub cached_sub imind] = update_boost_set(labels_sub, cached_sub, imind, labels_sub0, cached_sub0, imind0, new_learner);
      end

%     new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java_inf', [1:41 42 43 44 46]);
     new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java_inf');%, [1:41 42 43 44 46]);
     cached_scores = apply_weak_learner(cached_scores, new_learner);
      cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

      roc_50_t{boost_iter} = test_given_cache(D, cached_scores, cls, [0.5]);
      roc_50_test_t{boost_iter} = test_given_cache(Dtest, cached_scores_test, cls, [0.5]);
   end

