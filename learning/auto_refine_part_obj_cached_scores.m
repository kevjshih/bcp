function cached_scores = auto_refine_part_obj_cached_scores(VOCopts, part)
% this is necessary because we train automatically refined parts with ground-truth boxes
% if we want to train a boosted object classifier, we should use the region-based cached_scores

basedir = fullfile(VOCopts.localdir, 'auto_refined_models');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [part.name '.mat']);

if ~fileexists(cached_filename)
   disp(['Error: not trained yet']);
   return;
else
   disp(['Auto obj_cached_scores for part ' part.name]);
   load(cached_filename);

   if exist('obj_cached_scores', 'var')
      cached_scores = obj_cached_scores;
      return;
   else
      obj_cached_scores = compute_obj_cached_scores(model, w_loo);

      clear VOCopts;
      clear part;
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