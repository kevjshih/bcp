% javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

cls = 'cat';

if(1)
   load_init_data;
end

empty_model = model;

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

%%%%%%%%%%%%% Select parts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up model flags
model.hard_local = 0;
model.score_feat = 0;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but
                      %we want
model.do_boxes = 1;
new_incremental = 1;

% for i = 1:length(chosen)
for i = 1
%% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   best_model = candidate_models{chosen(i)};
   [Dsub cached_sub sub_im_inds] = collect_pos_subset(cls, D, cached_scores, pos_prec{chosen(i)}, 0.3);

%% Update model bookkeeping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   model = add_model(model, best_model); %, 1); % 1 indicates adding a spatial model
   chosen_names{i} = best_model.name;
   
   %[Dsub cached_sub] = get_exemplar_plus_neg(D, cached_scores, best_model, cls);
   %model = train_loo(model, Dsub, cached_sub, 7, 1, 1, 1e-2); % Train only on exemplar
   

%% Learn to use spatial features: %%%%%%%%%%%%%%%%%%%%%%%
   % Get strong initialization for spatial model: Use regions with best GT overlap 
%   cached_sub_loc = get_best_pos_reg(Dsub, cached_sub, cls);
%   [model w_loo w_noloo all_models neg_feat_cache] = train_loo(model, Dsub, cached_sub_loc, inf, 1, 1, 1e-2);

%   clear cached_sub_loc;
   % Now allow it to search over all regions for a few more iterations
   [model w_loo w_noloo all_models] = train_loo(model, Dsub, cached_sub, 10, 5, 1, 5.0);%, neg_feat_cache);
   w_all = repmat({w_noloo}, size(cached_scores));
   w_all(sub_im_inds) = w_loo;

   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!

   % Compute precision for each positive example
   [recall prec refined_aps(i) new_pos_prec] = test_part_detections_D(cls, D, cached_scores, i);
   
   % Find all positive examples with >30% precision
   [Dsub cached_sub sub_im_inds] = collect_pos_subset(cls, D, cached_scores, new_pos_prec, 0.3);

   % Manually verify boxes
   [ex_im ex_bbox im_list obj_box best_part_boxes best_scores] = get_exemplar_boxes(Dsub, cached_sub, cls, model, i);
   correct = ui_grid_detection_checker(ex_im, ex_bbox, im_list, ...
                                       obj_box, best_part_boxes, ...
                                       best_scores);

%%%%%%%%%%%%% Temporary %%%%%%%%%%%%%%%%%%%%%
if(0)
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);
   i = 2;
   [recall_tr{i} prec_tr{i} refined_tr_aps(i)] = test_part_detections_D(cls, D, cached_scores, i);
   [recall_te{i} prec_te{i} refined_te_aps(i)] = test_part_detections_D(cls, Dtest, cached_scores_test, i);
   model.part(2).computed = 1;

   [ex_im ex_bbox im_list obj_box best_part_boxes best_scores] = get_exemplar_boxes(D, cached_scores, cls, model, i);
   correct = ui_grid_detection_checker(ex_im, ex_bbox, im_list, obj_box, best_part_boxes, best_scores);

   correct_sub = find(correct);
   correct_sub = correct_sub(1:77);

   model.part(3) = model.part(2);
   model.part(3).computed = 0;
   model.num_parts = 3;

   [Dsub cached_sub sub_im_inds] = collect_manual_pos_subset(cls, D, cached_scores, im_list(correct_sub), obj_box(correct_sub), best_part_boxes(correct_sub));

   [model w_loo w_noloo all_models] = train_loo(model, Dsub, cached_sub, 10, 5, 1, 5.0);%, neg_feat_cache);
   w_all = repmat({w_noloo}, size(cached_scores));
   w_all(sub_im_inds) = w_loo;

   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);


   % Compute precision for each positive example
   i = 3;
   [recall_tr{i} prec_tr{i} refined_tr_aps(i)] = test_part_detections_D(cls, D, cached_scores, i);
   [recall_te{i} prec_te{i} refined_te_aps(i)] = test_part_detections_D(cls, Dtest, cached_scores_test, i);
end
%%%%%%%%%%%% Temporary %%%%%%%%%%%%%%%%%%%%%%%%
   % Prune boxes
   [Dsub cached_sub sub_im_inds] = collect_manual_pos_subset(cls, D, cached_scores, im_list(correct==1), obj_box(correct==1), best_part_boxes(correct==1));

   [model w_loo w_noloo all_models] = train_loo(model, Dsub, cached_sub, 10, 5, 1, 5.0);%, neg_feat_cache);
   w_all = repmat({w_noloo}, size(cached_scores));
   w_all(sub_im_inds) = w_loo;

   [labels cached_scores2] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
   [recall2 prec2 refined_aps2(i) new_pos_prec2] = test_part_detections_D(cls, D, cached_scores2, i);
%% Boosting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %[labels cached_scores] = collect_boost_data(model, D, cached_scores); % For comparing loo to non-loo
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
 
   save(fname, 'model', 'chosen_names', 'roc_50_test', 'roc_50', 'cached_scores', 'cached_scores_test');
end




return;
   i = 1;
   [recall1 prec1 refined_aps1(i) new_pos_prec1] = test_part_detections_D(cls, D, cached_scores1, i);
   [recall2 prec2 refined_aps2(i) new_pos_prec2] = test_part_detections_D(cls, D, cached_scores2, i);
