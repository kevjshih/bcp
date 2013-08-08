javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');


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

fname0 = fullfile('data/results', cls, 'improvement_search_%d.mat');

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

% Generate learning schedule.  This could be reordered based on current performance
% This should be precomputed and saved ...
[pos_prec chosen] = choose_candidate_amp(model, D, cached_scores, candidate_models);

% Set up local flags
subset = 0;

% Set up model flags
model.hard_local = 0;
model.score_feat = 0;

num_tests = 6;

for set_to_test = 1:6
   % Setup parameters for improvements to test:
   switch(set_to_test)
      case 1,
      % 1) baseline
         usegt = 0;
         model.do_transform = 0;
         do_spatial_model = 0;
         model.shift = 0;
      case 2,
      % 2) Use gt
         usegt = 1;
         model.do_transform = 0;
         do_spatial_model = 0;
         model.shift = 0;
      case 3,
      % 3) L/R flip
         usegt = 1;
         model.do_transform = 1;
         do_spatial_model = 0;
         model.shift = 0;
%      case 4,
%      % 4) Rotation
%         usegt = 1;
%         model.do_transform = 1;
%         model.shift = [0];
%         model.rotation = [-20 -10 0 10 20];
%         do_spatial_model = 0;
      case 4,
      % 4) Shift
         usegt = 1;
         model.do_transform = 1;
         model.shift = [0 4];
         do_spatial_model = 0;
      case 5,
      % 5) Spatial model
         usegt = 1;
         model.do_transform = 1;
         do_spatial_model = 1;
         model.shift = 0;
      case 6,
      % 6) Spatial model with shift
         usegt = 1;
         model.do_transform = 1;
         do_spatial_model = 1;
         model.shift = [0 4];
   end
   

   if(usegt || do_spatial_model)
      cached_gt = get_gt_pos_reg(D, cached_scores, cls);
   end


   for i = 1:2 % Test on 2 different parts
   %% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      best_model = candidate_models{chosen(i)};


      if(usegt && model.score_feat)
         cached_gt = get_gtbest_pos_reg(D, cached_scores, cls);
      end

      part_ind = (set_to_test-1)*2 + i;
      if(part_ind<=model.num_parts)
         continue;
      end
   
      fprintf('Training %d %d\n', set_to_test, i);
      %% Update model bookkeeping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      model = add_model(model, best_model, do_spatial_model); %, 1); % 1 indicates adding a spatial model

      if(isfield(model.part(part_ind), 'spat_const') && ~isempty(model.part(part_ind).spat_const) && model.part(part_ind).spat_const(5)>0.5) % Use FGMR criteria for objects
         model.part(part_ind).spat_const(5) = 0.5;
      end

      %% Learn to use spatial features: %%%%%%%%%%%%%%%%%%%%%%%
      if(do_spatial_model)
         [Dsub cached_gt_sub] = get_exemplar_plus_neg(D, cached_gt, best_model, cls);
         model = train_loo_cache(model, Dsub, cached_gt_sub, 7, 1, 1, 5.0); % Train only on exemplar
      end

      if(usegt)
         [model w_loo w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 5, 1, 5.0);%, neg_feat_cache);
         w_all = w_loo;
      else
         [model w_loo w_noloo all_models] = train_loo_cache(model, D, cached_scores, 10, 5, 1, 5.0);%, neg_feat_cache);
         w_all = w_loo;
      end


      % Compute Part Scores
      [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
      %[labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
      [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);
      model.part(part_ind).computed = 1;


      [recall_tr{set_to_test, i} prec_tr{set_to_test, i} refined_aps(set_to_test, i)] = test_part_detections_D(cls, D, cached_scores, part_ind);
      [recall_te{set_to_test, i} prec_te{set_to_test, i} refined_test_aps(set_to_test, i)] = test_part_detections_D(cls, Dtest, cached_scores_test, part_ind);

      save(fname, 'recall_*', 'prec_*', 'refined_*aps', 'model');
   end
end
