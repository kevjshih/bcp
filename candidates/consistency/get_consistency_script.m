cls = 'aeroplane';
load_init_data;
empty_model = model;

cached_gt = get_gtbest_pos_reg(D, cached_scores, cls);

%%%%%%%%%%%%%% Load candidate models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
candidate_file = fullfile('data', [cls '_candidates.mat']);
load(candidate_file, 'pos_prec', 'chosen', 'aps', 'candidate_models');


% Set up model flags
model.hard_local = 0;
model.score_feat = 0;
model.weighted = 1;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but we want


mind = chosen(1);
model = add_model(model, candidate_models{(mind)});
model.part(end).spat_const = [0 1 0.8 1 0 1];
%[cached_gt2 model.part.reference_box] = get_consistent_examples(model, D, cached_gt);
%model.part(1).spat_const = [0 1 0 1 0.8 1];

%[model2] = train_latent_whog(model, D, cached_gt2, 5);
%[model2_svm] = train_loo_cache(model, D, cached_gt2, 10, 5, 1, 15);

%model2.part(2) = model.part(1);
%model2.num_parts = 2;


model0 = model;
neg_feats = [];

model_tmp{1} = model0;

for i = 2:6
   %model_tmp{i-1}.part(1).spat_const = [0 1 0.8 1 0 1];
   %model_tmp{i-1}.part(1).reference_box = [];
   [cached_gt_tmp{i} bbox] = get_consistent_examples(model_tmp{i-1}, D, cached_gt, 1);
   %[cached_gt_tmp{i} model_tmp{i-1}.part.reference_box] = get_consistent_examples(model_tmp{i-1}, D, cached_gt);
   %model_tmp{i-1}.part(1).spat_const = [0 1 0 1 0.8 1];

   [model_tmp{i} neg_feats] = train_loo_cache(model_tmp{i-1}, D, cached_gt_tmp{i}, 10, 1, 1, 15, neg_feats);
end


model2 = model0;
for i = 2:length(model_tmp)
   model2.part(i) = model_tmp{i}.part;
   model2.part(i).spat_const = [0 1 0.8 1 0 1];
%   model2.part(i).reference_box = [];
   model2.num_parts = i;
end


[dk cached_scores_test] = collect_boost_data(model2, Dtest, cached_scores_test);

for i = 1:model2.num_parts
   i
   [recall_full{i}, prec_full{i}, ap_full(i)] = test_part_detections_D(cls, Dtest, cached_scores_test, i);
end






[model_full neg_feat] = train_loo_cache(model, D, cached_gt, 10, 2, 0.5, 15);
[model_full] = train_loo_cache(model_full, D, cached_gt, 10, 3, 1, 15, neg_feat);
