function refinement_round = manual_refine_part(VOCopts, part, skip_refine)
% Returns the data for every round of refinement and optionally performs an additional round of refinement.
%
% Only retrieves previous refinement data (doesn't refine the part
% again) if 'skip_refine' is true.
%
% The first call only initializes the part but is fully automatic
% unless applicability hasn't been set yet.

if ~exist('skip_refine', 'var')
   skip_refine = false;
end

basedir = fullfile(VOCopts.localdir, 'manual_refinement_rounds');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [part.name '.mat']);
num_rounds_cached_filename = fullfile(basedir, [part.name '_num_rounds.mat']);

if ~fileexists(cached_filename)
   disp('Initializing manual refinement training...');
   %% Initialize training.
   cls = part.cls;
   set = part.set;

   load_init_data;

   % Set up model flags
   model.hard_local = 0;
   model.score_feat = 0;
   model.incremental_feat = 0;
   model.do_transform = 1;
   model.shift = [0];
   model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but
                         %we want
   model.do_boxes = 1;

   % Get the applicability of the part to the objects in the set.
   applicability = manual_set_applicability(VOCopts, part, 'train', true);

   % Initialize the part model.
   part_model = orig_train_exemplar(VOCopts, part.im, part.bbox, cls, set, true);
   part_model.model.name = part_model.models_name;
   model = add_model(model, part_model.model);

   % Automatically refine the initial part model with 10 automatically-chosen examples.
   cached_scores = get_gtbest_pos_reg(D, cached_scores, cls);  % Use object ground truth boxes
   [model dc w_loo w_noloo all_models] = train_loo_cache(model, D, cached_scores, 10, 2, 10, 15.0);
   w_all = w_loo;

   % Compute Part Scores
   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all);

   round_i = 1;
else
   fprintf(['Loading refinement rounds from "' cached_filename '"...\n']);
   if skip_refine
      disp('Refinement skipped');
      load(num_rounds_cached_filename, 'round_i');
      round_cached_filename = fullfile(VOCopts.localdir, 'manual_refinement_rounds', [part.name ' round ' num2str(round_i - 1) '.mat']);

      load(round_cached_filename);
      return;
   end

   load(cached_filename);

   disp(['Manually refining round ' num2str(round_i) '...']);
   round_cached_filename = fullfile(VOCopts.localdir, 'manual_refinement_rounds', [part.name ' round ' num2str(round_i) '.mat']);

   %% Obtain refinement training examples.
   % Compute precision for each positive example
   [recall prec refined_aps(1) new_pos_prec] = test_part_detections_D(cls, D, cached_scores, 1);

   % Prune positive examples so the user doesn't have to flip through so many.
   [Dsub cached_sub sub_im_inds] = collect_pos_subset(cls, D, cached_scores, new_pos_prec, -inf);

   % Manually verify the training detections.
   [dc dc dc im_names obj_box obj_box0 best_part_boxes best_scores] = get_exemplar_boxes(Dsub, cached_sub, cls, model, 1);

   % Check if the image is flipped (indices where obj_box/obj_box0 are not the same).
   flipped = zeros(length(obj_box), 1);
   for obj_box_i = 1:length(obj_box)
      if all(obj_box{obj_box_i} == obj_box0{obj_box_i})
         flipped(obj_box_i) = 0;
      else
         flipped(obj_box_i) = 1;
      end
   end

   BDglobals;
   ims = cell(length(im_names), 1);
   for ims_i = 1:length(ims)
      ims{ims_i} = convert_to_I(fullfile(im_dir, im_names{ims_i}));
      if flipped(ims_i)
         ims{ims_i} = ims{ims_i}(:, end:-1:1, :);
      end
   end
   correct = ui_check_part(part, ims, obj_box, ...
                           manual_get_applicability(applicability, im_names, obj_box), ...
                           best_part_boxes, best_scores);
   clear ims;  % Save space by not saving all the images
   % Find all positive examples that the user marked correct.
   [Dsub cached_sub sub_im_inds] = collect_manual_pos_subset(cls, D, cached_scores, im_names(correct==1), obj_box0(correct==1), best_part_boxes(correct==1));

   %% Refine with the training examples obtained above.
   [model dc w_loo w_noloo all_models] = train_loo_cache(model, Dsub, cached_sub, 10, 2, 1, 15.0);
   w_all = repmat({w_noloo}, size(cached_scores));
   w_all(sub_im_inds) = w_loo;

   [labels cached_scores] = collect_boost_data_loo(model, D, cached_scores, w_all);
   %[recall2 prec2 refined_aps2(1) new_pos_prec2] = test_part_detections_D(cls, D, cached_scores, 1);

   [model.part.computed] = deal(0);

   %% Save the data for the current round of refinement.
   refinement_round.model = model;

   refinement_round.refine_im_names = im_names;
   refinement_round.refine_flipped = flipped;
   refinement_round.refine_obj_box = obj_box;
   refinement_round.refine_best_part_boxes = best_part_boxes;
   refinement_round.refine_best_scores = best_scores;
   refinement_round.refine_correct = correct;

   %refinement_round.w_all = w_all;

   save(round_cached_filename, 'refinement_round');
   clear refinement_round;

   round_i = round_i + 1;
end

clear VOCopts;
clear part;
clear skip_refine;
save(cached_filename, '-v7.3');
save(num_rounds_cached_filename, '-v7.3', 'round_i');
end