function refinement_round = manual_eval_manual_refine_part_round(VOCopts, part, round, skip_eval)
% Returns the correctness of the part on each object in the test
% set for the given round of refinement.
%
% The first call initializes 'correct' to all "Not Sure" values
% for applicable parts. It takes the longest because it has to
% compute everything from scratch, but it's fully automatic.
%
% If 'skip_eval' is true, we retrieve the evaluation data from a
% previous run if possible.

if ~exist('skip_eval', 'var')
   skip_eval = false;
end

basedir = fullfile(VOCopts.localdir, 'manual_refinement_rounds');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [part.name ' round ' num2str(round) '.mat']);

% Check if the part has been refined at all yet.
if ~fileexists(cached_filename)
   fprintf(['Error: part ' part.name ' has not been refined ' num2str(round) ' times yet.\n']);
   return;
end

fprintf(['Loading refinement round from "' cached_filename '"...\n']);
load(cached_filename);

eval_model = refinement_round.model;

% Check if the user wants to skip evaluation and then return the
% requested data if it exists.
if skip_eval && isfield(refinement_round, 'eval_correct') && ~isempty(refinement_round.eval_correct) && isfield(refinement_round, 'eval_im_names') && (numel(find(refinement_round.eval_correct == 1)) > 0 || numel(find(refinement_round.eval_correct == -1)) > 0)
   correct = refinement_round.eval_correct;
   return;
end

if ~isfield(refinement_round, 'eval_correct') || isempty(refinement_round.eval_correct) || ~isfield(refinement_round, 'eval_im_names')
   % If this refinement round hasn't been evaluated before,
   % initialize the evaluation.
   cls = part.cls;
   load_init_data;

   cached_scores_test = get_gtbest_pos_reg(Dtest, cached_scores_test, cls);  % Use object ground-truth boxes
   [dc eval_cached_scores_test] = collect_boost_data(eval_model, Dtest, cached_scores_test);
   [dc dc im_list_test im_names_test obj_box_test obj_box0_test best_part_boxes_test best_scores_test] = get_exemplar_boxes(Dtest, eval_cached_scores_test, cls, eval_model, 1);
   applicability_test = manual_set_applicability(VOCopts, part, 'test', true);
   correct_test = manual_get_applicability(applicability_test, im_names_test, obj_box_test);

   % Check if the image is flipped (indices where obj_box/obj_box0 are not the same).
   flipped = zeros(length(obj_box_test), 1);
   for obj_box_test_i = 1:length(obj_box_test)
      if all(obj_box_test{obj_box_test_i} == obj_box0_test{obj_box_test_i})
         flipped(obj_box_test_i) = 0;
      else
         flipped(obj_box_test_i) = 1;
      end
   end
else
   % This refinement round has been evaluated before, so reload
   % those values instead of recomputing.
   im_names_test = refinement_round.eval_im_names;
   flipped = refinement_round.eval_flipped;
   obj_box_test = refinement_round.eval_obj_box;
   best_part_boxes_test = refinement_round.eval_best_part_boxes;
   best_scores_test = refinement_round.eval_best_scores;
   correct_test = refinement_round.eval_correct;

   % Manually evaluate the data.
   BDglobals;  % Get the proper 'im_dir'.
   ims = cell(length(im_names_test), 1);
   for ims_i = 1:length(ims)
      ims{ims_i} = convert_to_I(fullfile(im_dir, im_names_test{ims_i}));
      if flipped(ims_i)
         ims{ims_i} = ims{ims_i}(:, end:-1:1, :);
      end
   end
   correct_test = ui_check_part(part, ims, obj_box_test, correct_test, ...
                                best_part_boxes_test, best_scores_test);
   clear ims;
end

% Record evaluation data.
refinement_round.eval_im_names = im_names_test;
refinement_round.eval_flipped = flipped;
refinement_round.eval_obj_box = obj_box_test;
refinement_round.eval_best_part_boxes = best_part_boxes_test;
refinement_round.eval_best_scores = best_scores_test;
refinement_round.eval_correct = correct_test;

% Clear input parameters and save data.
clear VOCopts;
clear part;
clear round;
clear skip_eval;
clear cached_scores_test;
clear eval_cached_scores_test;
save(cached_filename, '-v7.3', 'refinement_round', 'im_dir');

correct = correct_test;
end