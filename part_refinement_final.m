function part_refinement_final(cls)


DO_TRAIN_BOOSTING_AND_EVALUATE = 1

num_parts_used = 40; % number of parts to use per category


startup;


% Parse input
try, IGNORE_SIMILAR = str2num(IGNORE_SIMILAR); end
try, if(~isinf(str2num(cls))), cls=str2num(cls); end, end


%%% Setup class...
BDglobals;

BDpascal_init;
VOCopts.sbin = 8;
VOCopts.localdir = fullfile(WORKDIR, 'exemplars');

if(isnumeric(cls))
    clsind = cls;
    cls = VOCopts.classes{clsind};
end


fprintf('Doing category: %s\n', cls);
%%%%%%%%%%%%%%%%%%%%%%%%%

if(~exist('IGNORE_SIMILAR', 'var'))
    IGNORE_SIMILAR = 0;
end


% TRAINSET variable set in BDglobals
trainval = strcmp(TRAINSET, 'trainval');
fprintf('Training on %s split\n', TRAINSET);

if(trainval)
    set_str = 'trainval';
else
    set_str = 'train';
end


if(IGNORE_SIMILAR)
    refinement_type = 'final_nosimilar';
else
    refinement_type = 'final';
end

base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));

if(~exist(base_dir, 'file'))
    mkdir(base_dir);
end

if(trainval)
    set_str = 'trainval';
    C = 2*15; % Using twice as many examples

    load_init_final;
else % Just the training set
    set_str = 'train';
    C  = 15;
    %candidate_file = fullfile('data', [cls '_candidates_whog.mat']);

    load_init_data;
    %clear Dtest cached_scores_test;
end
empty_model = model;


%%%%%%%%%%%%%% Load models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
candidate_file = fullfile('data/tmp/candidates/', [cls '_' set_str '_candidates.mat']);

if(~exist(candidate_file, 'file'))
    candidate_suffix = 'whog';

    train_candidate_parts(cls, 2000, candidate_suffix, trainval);
    test_all_candidates(10, 10, cls, candidate_suffix, trainval);

    candidate_suffix_full = [candidate_suffix '_' set_str];
    candidate_models = load_candidate_models(cls, 0, 0, candidate_suffix_full); % Use auto selected ones for now
    candidate_models = [candidate_models, load_candidate_models(cls, 0, 1, candidate_suffix_full)]; % Add object level 
    
    % Generate learning schedule.  This could be reordered based on current performance
    [pos_prec chosen aps] = choose_candidate_amp(model, D, cached_scores, candidate_models);
    save(candidate_file, 'pos_prec', 'chosen', 'aps', 'candidate_models');
else
    load(candidate_file, 'pos_prec', 'chosen', 'aps', 'candidate_models');
end

%%%%%%%%%%%%%% Resume
fname = fullfile(base_dir, 'part_detections.mat');
resfname = fullfile(base_dir, 'trained_classifier.mat');

if(exist(fname, 'file'))
    fprintf('Resuming!\n');
    load(fname);
    fprintf('%d parts already finished\n', model.num_parts);
end
%%%%%%%%%%%%% Select parts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



incremental_selection = 0;


if(IGNORE_SIMILAR)
    similar = get_similar_images(D, cls);

    if(length(D)==length(cached_scores)) % If lengths don't match, similar examples have already been removed
        cached_scores(similar) = [];
    end

    D(similar) = [];
end

% Set up model flags
model.hard_local = 0;
model.score_feat = 0;
model.weighted = 0;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but we want
model.cached_weight = 0;
% Consistency thresholds...
model.min_ov = 0.75;
model.min_prob = 0.3;

cached_gt = get_gt_pos_reg(D, cached_scores, cls);

[Dpos pos_ind] = LMquery(D, 'object.name', cls, 'exact');
if(~exist('cached_gt_pos', 'var'))
    cached_gt_pos = cached_gt(pos_ind);
end

if(model.num_parts>0)
    [model.part.computed] = deal(1);
end

% refine each part
for i = 1:num_parts_used
    iter = i;

    if(i<=model.num_parts)
        fprintf('Part %d is already done\n', i);
        continue;
    end

    %% Select data subset, etc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(incremental_selection==1)
        [next_pos_prec chosen_t] = choose_next_candidate_amp(model, D, cached_gt, candidate_models(not_chosen==1), pos_prec(not_chosen==1));
        not_chosen_ind = find(not_chosen);
        chosen(i) = not_chosen_ind(chosen_t);
        not_chosen(chosen(i)) = 0;
        best_model = candidate_models{chosen(i)};
    else
        best_model = candidate_models{chosen(i)};
    end

    model = add_model(model, best_model); %, 1); % 1 indicates adding a spatial model
    model.part(iter).bias = model.part(iter).bias + 0.5; % Make sure to pull in plenty of hard negatives in the first iteration
    model.part(iter).spat_const = [0 1 0.8 1 0 1];
    chosen_names{iter} = best_model.name;

    try
    % part model refinement done here
    [model w_all] = train_consistency(model, D, cached_gt);
    catch
     w_all = []; % Model failed to train!
    end
    % Compute Part Scores to be used with the boosting
    [labels cached_gt_pos] = collect_boost_data_loo(model, Dpos, cached_gt_pos, w_all); 
    [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all); 
    w_all_storage{iter} = w_all;
    model.part(iter).computed = 1;
    
    % save progress incrementally
    save(fname, '-v7.3', 'model', 'chosen_names', 'cached_gt_pos', 'cached_scores', 'w_all_storage');
end

computed = get_parts_computed(model, cached_scores_test);

if trainval
    % load the test meta data
    load_init_test
end

if(isempty(computed) || any(~computed))
    [model.part.computed] = deal(0);
    [model.part(computed).computed] = deal(1);
    [labels_test cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test); % We'll do this at the end!
    [model.part.computed] = deal(1);
    save(fname, '-v7.3', 'model', 'chosen_names', 'cached_gt_pos', 'cached_scores', 'cached_scores_test', 'w_all_storage');
else
    fprintf('Test set detections already computed\n');
end


% Train localization model
if DO_TRAIN_BOOSTING_AND_EVALUATE

    calib_params = train_detector_calibration(D, cached_scores, cls);
    [model.part.calib] = deal(calib_params{:});

    cached_pos = cached_scores(pos_ind);
    % train the relocalization model
    model_wbox = train_bbox_new(Dpos, cached_pos, model, calib_params);

    % use model to compute new boxes with model
    cached_scores_box = box_inference(cached_scores, model_wbox);

    % train boosted classifier using new boxes
    model_wbox.learner = boost_iterate(D, cached_scores_box, cls, 5, 'sigmoid_java');

    % rectify test/val set bounding boxes
    cached_scores_test_box = box_inference(cached_scores_test, model_wbox);

    % apply boosted classifier
    cached_scores_test_box = apply_weak_learner(cached_scores_test_box, model_wbox.learner);

    if ~trainval
        % evaluate on val data
        res = test_given_cache(Dtest, cached_scores_test_box, cls, 0.5);
        
        % save the relocalization models, boosted classifier, and results on val set
        save(resfname, '-v7.3', 'model_wbox', 'res'); 
    else
        save(resfname, '-v7.3', 'model_wbox'); 
    end

end






