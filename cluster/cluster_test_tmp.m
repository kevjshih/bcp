function cluster_test(cls, trainval, refinement_type, Ntodo)

% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
end

try
   trainval = str2num(trainval);
end

matlabpool_robust;

%%%%%%%%%%%%%%%%% Setup class... %%%%%%%%%%%%%%%%%%%%%%%%%%
startup_cluster
BDglobals;

VOCinit;
VOCopts.sbin = 8;
VOCopts.localdir = fullfile(WORKDIR, 'exemplars');

if(isnumeric(cls))
   clsind = cls;
   cls = VOCopts.classes{clsind};
end

fprintf('Doing category: %s\n', cls);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(trainval)
   set_str = 'trainval';
   load_init_final;
else % Just the training set
   set_str = 'train';
   load_init_data;
end


base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
result_str = fullfile(base_dir, sprintf('part_detections.mat'));

load(result_str, 'model', 'cached_scores', 'cached_gt_pos', 'cached_scores_test', 'cached_gt_pos_test');


[model.part.computed] = deal(0);
[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');
cached_pos_noloo = cached_scores(pos_inds);
[dk cached_pos_noloo] = collect_boost_data(model, Dpos, cached_pos_noloo);
[model.part.computed] = deal(1);

save(result_str, 'model', 'cached_scores', 'cached_gt_pos', 'cached_scores_test', 'cached_gt_pos_test', 'cached_pos_noloo');
