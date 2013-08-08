set = 'train';
cls = 'aeroplane';
part_inds = 1:4;
rounds = [3, 4, 3, 4];

load_init_data;

%% Retrieve the parts
all_parts = get_manual_parts(VOCopts, set, cls);
parts = {};
for part_ind = part_inds
   parts{end+1} = all_parts{part_ind};
end

%% Get the combined object part models
tic;
manual_model = {};
manual_cached_scores = {};
for i = 1:length(parts)
   part = parts{i};
   round = rounds(i);

   round = manual_refine_part_round(VOCopts, part, round);
   manual_model{end+1} = round.model;
   if isempty(round.obj_cached_scores)
      manual_cached_scores{end+1} = manual_refine_part_round_obj_cached_scores(VOCopts, part, round);
   else
      manual_cached_scores{end+1} = round.obj_cached_scores;
   end
end

[manual_model, manual_cached_scores] = combine_part_models(manual_model, manual_cached_scores);
disp('combine manual part models');
toc;

%% Train the boosted detector
tic;
[manual_model.part.computed] = deal(0);
manual_new_learner = boost_iterate(D, manual_cached_scores, cls, 5, 'sigmoid_java');

[dc manual_cached_scores_test] = collect_boost_data(manual_model, Dtest, cached_scores_test);
manual_cached_scores_test = apply_weak_learner(manual_cached_scores_test, manual_new_learner);
disp('train manual boosted detector');
toc;

save('manual_cached_scores_test.mat', 'manual_cached_scores_test');
clear manual_model;
clear manual_cached_scores;
clear manual_new_learner;
clear manual_cached_scores_test;

%% Compute Statistics
figure;
manual_roc_50 = test_given_cache(Dtest, manual_cached_scores_test, cls, [0.5]);

% Computing P-R curve for each part
figure;
hold all;
for part_ind = 1:length(parts)
   [manual_recall, manual_prec, manual_ap] = test_part_detections_D(cls, Dtest, manual_cached_scores_test, part_ind);
   plot(manual_recall, manual_prec);
end
title('manual p-r curve');
hold off;

%% Get the combined part models
tic;
auto_model = {};
auto_cached_scores = {};
for i = 1:length(parts)
   part = parts{i};

   auto_model{end+1} = auto_refine_part(VOCopts, part);
   auto_cached_scores{end+1} = auto_refine_part_obj_cached_scores(VOCopts, part);
end

[auto_model auto_cached_scores] = combine_part_models(auto_model, auto_cached_scores);
disp('combine auto part models');
toc;

%% Train the boosted detector
tic;
[auto_model.part.computed] = deal(0);
auto_new_learner = boost_iterate(D, auto_cached_scores, cls, 5, 'sigmoid_java');

[dc auto_cached_scores_test] = collect_boost_data(auto_model, Dtest, cached_scores_test);
auto_cached_scores_test = apply_weak_learner(auto_cached_scores_test, auto_new_learner);
disp('train auto boosted detector');
toc;

save('auto_cached_scores_test.mat', 'auto_cached_scores_test');
clear auto_model;
clear auto_cached_scores;
clear auto_new_learner;
clear auto_cached_scores_test;

%% Compute Statistics
figure;
auto_roc_50 = test_given_cache(Dtest, auto_cached_scores_test, cls, [0.5]);

% Computing P-R curve for each part
figure;
hold all;
for part_ind = 1:length(parts)
   [auto_recall, auto_prec, auto_ap] = test_part_detections_D(cls, Dtest, auto_cached_scores_test, part_ind);
   plot(auto_recall, auto_prec);
end
title('auto p-r curve');
hold off;