javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

% cls = 'object';

load_init_data;
empty_model = model;

%%%%%%%%%%%%%% Create initial model %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Get filename (we want to avoid overwriting previous results) %%%%%%%%%
if(~exist(fullfile('data/results', cls), 'file'))
   mkdir(fullfile('data/results', cls));
end

fname0 = fullfile('data/results', cls, 'incremental_model_rand_object_%d.mat');

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


% Set up local flags
%subset_predefined = 0;
%subset_search = 0;
%usegt = 1; % This flag isn't used anymore, everything is trained with GT
subsetonly = 1;
subsetinit = 0;
do_spatial_model = 0;
incremental_selection = 0;


% Set up model flags
model.hard_local = 0;
model.score_feat = 0;
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
   if(mod(i,4)==1) % Use full object template
      best_model = train_candidate_parts(cls, 1);
   else
      best_model = train_candidate_parts(cls, 1, ); % TODO: finish this!

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
      [model neg_feats w_all w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 5, 0.5, 5.0);%, neg_feat_cache);
   elseif(subsetinit)
      [model neg_feats] = train_loo_cache(model, D, cached_gt, 10, 2, 0.5, 5.0); % Strong initialization with subset of the data
      [model neg_feats w_all w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 3, 1, 5.0, neg_feats); % Refine with the rest to pick up any other examples
   else 
      [model neg_feats w_all w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 5, 1, 5.0);%, neg_feat_cache);
   end
   
   clear neg_feats
%% Boosting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
   %[labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);
   model.part(i).computed = 1;

   %[labels_sub cached_sub] = prune_boost_data(D, cached_scores, cls); % Latent search doesn't seem to be necessary
   [labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
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

