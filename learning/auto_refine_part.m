function [model, num_examples] = auto_refine_part(VOCopts, part, use_cached)
if ~exist('use_cached', 'var')
   use_cached = true;
end
BDglobals;
basedir = fullfile(VOCopts.localdir, 'auto_refined_models');
if ~exist(basedir, 'dir')
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [part.name '.mat']);
num_examples_filename = fullfile(basedir, [part.name 'num_examples.mat']);

num_examples = -1;
if ~fileexists(cached_filename) || ~use_cached
   cls = part.cls;
   set = part.set;

   if strcmp(TRAINSET, 'train')
       load_init_data;
   else
       load_init_final;
   end

   %% Set up model flags
   model.hard_local = 0;
   model.score_feat = 0;
   model.incremental_feat = 0;
   model.do_transform = 1;
   model.shift = [0];
   model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but
                      %we want
   model.do_boxes = 1;

   part_model = orig_train_exemplar(VOCopts, part.im, part.bbox, cls, set, true);
   part_model.model.name = part_model.models_name;

   model = add_model(model, part_model.model);

   %% Refine completely automatically.
   cached_scores = get_gtbest_pos_reg(D, cached_scores, cls);  % Use object ground-truth boxes
   [model dc w_loo w_noloo all_models] = train_loo_cache(model, D, cached_scores, 10, 2, 1, 15.0);
   num_examples = length(w_loo);

   save(cached_filename, '-v7.3', 'model', 'num_examples');
   save(num_examples_filename, 'num_examples');
else
   load(cached_filename, 'model', 'num_examples');
   load(num_examples_filename, 'num_examples');
end
end