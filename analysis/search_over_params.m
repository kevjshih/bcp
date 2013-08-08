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

%%%%%%% Get filename (we want to avoid overwriting previous results) %%%%%%%%%
if(~exist(fullfile('data/results', cls), 'file'))
   mkdir(fullfile('data/results', cls));
end

fname0 = fullfile('data/results', cls, 'refinement_param_search_%d.mat');

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
%[pos_prec chosen init_aps] = choose_candidate_amp(model, D, cached_scores, candidate_models);

%exemplar_aps = init_aps(chosen);
subset = 0;

subset_predefined = 0;
subset_search = 0;
usegt = 1;
do_spatial_model = 0;


% Set up model flags
model.hard_local = 0;
model.score_feat = 0;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but we want

if(usegt)
   cached_gt = get_gt_pos_reg(D, cached_scores, cls);
end

%Cs = 5.^[-3 -2 -1 1 2 3];
%Cs = 5.^[0 1 2 3];
Cs = 5.^[-7:-2];
Ws = 1;%[5 1 10 20];
SSs = 1;%[0.25 0.5 0.75 1];

if(0)
% Subsample Dataset
[dk posinds] = LMquery(D, 'object.name', cls, 'exact');
neginds = 1:length(D); neginds(posinds) = [];
r = 1:length(neginds); %randperm(length(neginds));
neginds = neginds(r(1:min(end,500)));
D = D([posinds(:); neginds(:)]);
cached_scores = cached_scores([posinds(:); neginds(:)]);
cached_scores_noloo = cached_scores;

for i = 1:length(pos_prec)
   pos_prec{i} = pos_prec{i}([posinds(:); neginds(:)]);
end   


% Subsample testset too
[dk posinds] = LMquery(Dtest, 'object.name', cls, 'exact');
neginds = 1:length(Dtest); neginds(posinds) = [];
r = 1:length(neginds); %randperm(length(neginds));
neginds = neginds(r(1:min(end,500)));
Dtest = Dtest([posinds(:); neginds(:)]);
cached_scores_test = cached_scores_test([posinds(:); neginds(:)]);
end % if(0) ... subsample


for i = 1:3%length(chosen)
   for Sind = 1:length(SSs)
   for Cind = 1:length(Cs)
   for Wind = 1:length(Ws)
      C = Cs(Cind);
      S = SSs(Sind);
      W = Ws(Wind);

      part_i = sub2ind([length(Ws), length(Cs) length(SSs) length(chosen)], Wind, Cind, Sind, i);

      if(part_i<=model.num_parts) % Already computed
         fprintf('Skipping %d\n', part_i);
         continue;
      end
      %% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      best_model = candidate_models{chosen(i)};

      %% Update model bookkeeping %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      model = add_model(model, best_model); %, 1); % 1 indicates adding a spatial model
      model.part(part_i).C = C;
      chosen_names{part_i} = best_model.name;
   
      %% Learn to use spatial features: %%%%%%%%%%%%%%%%%%%%%%%
      if(isfield(model.part(part_i), 'spat_const') && ~isempty(model.part(part_i).spat_const) && model.part(part_i).spat_const(5)>0.5) % Use FGMR criteria for objects
         model.part(part_i).spat_const(5) = 0.5;
      end

      if(subset)
         [Dsub cached_sub sub_im_inds] = collect_pos_subset(cls, D, cached_scores, pos_prec{chosen(i)}, S);
         if(~isempty(sub_im_inds))
            [model w_loo w_noloo] = train_loo_cache(model, Dsub, cached_sub, 10, 5, 1, C);%, neg_feat_cache);
            w_all = repmat({w_noloo}, size(cached_scores));
            w_all(sub_im_inds) = w_loo;
         else
            w_all = repmat({get_model_weights(model)}, size(cached_scores));
         end
      else
         %[model w_loo w_noloo] = train_loo_cache(model, D, cached_scores, 10, 5, S, [C C*W]);%, neg_feat_cache);
%         [model w_loo w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 5, S, [C C*W]);%, neg_feat_cache);
         [model] = train_loo_cache(model, D, cached_gt, 10, 5, S, [C C*W]);%, neg_feat_cache);
   %      w_all = repmat({w_noloo}, size(cached_scores));
   %      w_all(sub_im_inds) = w_loo;
         w_all = w_loo;
      end
      %% Boosting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Compute Part Scores
      %[dk cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); % TODO!
      %[dk cached_scores_noloo] = collect_boost_data(model, D, cached_scores_noloo);
      [dk cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);

      model.part(part_i).computed = 1;

      %[recall_tr_loo{i, Sind, Cind, Wind} prec_tr_loo{i, Sind, Cind, Wind} refined_aps(i, Sind, Cind, Wind)] = test_part_detections_D(cls, D, cached_scores, part_i);
      %[recall_tr{i, Sind, Cind} prec_tr{i, Sind, Cind} refined_aps_noloo(i, Sind, Cind)] = test_part_detections_D(cls, D, cached_scores_noloo, part_i);
      [recall_te{i, Sind, Cind, Wind} prec_te{i, Sind, Cind, Wind} refined_test_aps(i, Sind, Cind, Wind)] = test_part_detections_D(cls, Dtest, cached_scores_test, part_i); 
      save(fname, 'model', 'chosen_names', 'refined_*', 'Cs', 'SSs', 'recall_*', 'prec_*', 'subset', 'Ws');
   end % Ws
   end % Cs
   end % SSs
end
