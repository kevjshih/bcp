function cached_scores = manual_refine_part_round_obj_cached_scores(VOCopts, part, round)
% this is necessary because we train manually refined parts with ground-truth boxes
% if we want to train a boosted object classifier, we should use the region-based cached_scores

basedir = fullfile(VOCopts.localdir, 'manual_refinement_rounds');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [part.name ' round ' num2str(round) '.mat']);

if ~fileexists(cached_filename)
   disp(['Error: not refined ' num2str(round) 'times yet']);
   return;
else
   load(cached_filename);

   disp(['Manual obj_cached_scores for part ' part.name ', round ' num2str(round)]);

   if isfield(refinement_round, 'obj_cached_scores') && ~isempty(refinement_round.obj_cached_scores)
      cached_scores = refinement_round.obj_cached_scores;
      return;
   else
      obj_cached_scores = compute_obj_cached_scores(refinement_round.model, refinement_round.w_all);
      refinement_round.obj_cached_scores = obj_cached_scores;

      clear VOCopts;
      clear part;
      clear round;
      save(cached_filename, '-v7.3');

      cached_scores = obj_cached_scores;
      return;
   end
end
end

function cached_scores = compute_obj_cached_scores(part_model, part_w_all)
% this functionality is in a separate function so it doesn't overwrite values
cls = part_model.cls;
load_init_data;
[labels cached_scores] = collect_boost_data_loo(part_model, D, cached_scores, part_w_all);
end