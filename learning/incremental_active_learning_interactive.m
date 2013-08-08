javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

cls = 'cat';

if(1)
load_init_data;
end

empty_model = model;

%%%%%%%%%%%%%% Load models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

candidate_models = load_candidate_models(cls, 1); % Use user selected ones for now

if(isempty(candidate_models))
   % Haven't trained them yet!!
   train_candidate_parts(cls, 1000); 
   candidate_models = load_candidate_models(cls);
%   cluster_test_candidates(cls, 10);
end

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
VOCinit;
VOCopts.sbin = 8;

% Get initial objects to select parts from.
ps = get_pos_stream(VOCopts, VOCopts.trainset, cls);
ims = {};
obj_box = {};
for i = 1:length(ps)
    for j = 1:size(ps{i}.bbox, 1)
        ims{end+1} = ps{i}.I;
        obj_box{end+1} = ps{i}.bbox(j, :);
    end
end

% Select parts from objects in training set.
parts = manually_select_parts_fancy(ims, obj_box);
I = cell(length(parts), 1);
bbox = cell(length(parts), 1);
for i = 1:length(parts)
    I{i} = parts{i}.im;
    bbox{i} = parts{i}.bbox;
end

% Train models from selected parts and convert format for compatibility.
candidate_models = orig_train_exemplar(VOCopts, I, bbox, cls, VOCopts.trainset, true);
if(~iscell(candidate_models))
   candidate_models = {candidate_models};
end

for i = 1:length(candidate_models)
    candidate_models{i}.model.name = [candidate_models{i}.models_name '.mat'];
    candidate_models{i} = candidate_models{i}.model;
end

model.hard_local = 0;
model.score_feat = 0;

% Generate learning schedule.  This could be reordered based on current performance
% This should be precomputed and saved ...
chosen = 1:length(candidate_models);

new_incremental = 1;

cached_gt = get_gt_pos_reg(D, cached_scores, cls);

for i = 1:length(chosen)
%% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   best_model = candidate_models{chosen(i)};

%% Update model bookkeeping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   model = add_model(model, best_model); %, 1); % 1 indicates adding a spatial model
   model.part(i).bias = model.part(i).bias + 0.5; % Make sure to pull in plenty of hard negatives in the first iteration
   chosen_names{i} = best_model.name;
   
   [model , ~, w_all w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 5, 1, 5.0);%, neg_feat_cache);

   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
   [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);
   model.part(i).computed = 1;

% Train Boosted model
   [labels_sub cached_sub] = prune_boost_data_overlap(D, cached_scores, cls);
   new_learner = boost_train(cached_sub, labels_sub, 'sigmoid_java');

% Test it!
   cached_scores = apply_weak_learner(cached_scores, new_learner);
   cached_scores_test = apply_weak_learner(cached_scores_test, new_learner);

   roc_50{i} = test_given_cache(D, cached_scores, cls, 0.5);
   roc_50_test{i} = test_given_cache(Dtest, cached_scores_test, cls, 0.5,0,0);
 
   save(fname, '-v7.3', 'model', 'chosen_names', 'roc_50*','cached_scores', 'cached_scores_test');
end




return;
   i = 1;
   [recall1 prec1 refined_aps1(i) new_pos_prec1] = test_part_detections_D(cls, D, cached_scores1, i);
   [recall2 prec2 refined_aps2(i) new_pos_prec2] = test_part_detections_D(cls, D, cached_scores2, i);
