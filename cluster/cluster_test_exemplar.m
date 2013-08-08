function cluster_test(cls, trainval, Ntodo)

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
   C = 2*15; % Using twice as many examples
   candidate_file = fullfile('data', [cls '_candidates_trainval.mat']);

   load_init_final;
else % Just the training set
   set_str = 'train';
   C  = 15;
   %candidate_file = fullfile('data', [cls '_candidates_whog.mat']);
   
   load_init_data;
   
   candidate_file = fullfile('data', [cls '_candidates_trainval.mat']);
end

load(candidate_file);
% Make sure the same parts are used for all cases, consistent is most constrainted, so all others should be supersets of it
consistent_base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, 'consistent'));

refinement_type = 'exemplar';
base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));

if(~exist(base_dir, 'file'))
   mkdir(base_dir);
end

cached_gt = get_gtbest_pos_reg(D, cached_scores, cls);
[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');

cached_gt_pos = cached_gt(pos_inds);

%%%%%%%%%%%%%%%% Do it! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.hard_local = 0;
model.score_feat = 0;
model.weighted = 0;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20];
model.cached_weight = 0;

num_parts = 0;

for i = 1:Ntodo
   consistent_fstr = fullfile(consistent_base_dir, sprintf('part_model_%d.mat', i));

   if(~exist(consistent_fstr, 'file'))
      fprintf('Part %d (%s) wasn''t found for consistent model!\n', i, consistent_fstr);
      continue;
   end
   num_parts = num_parts + 1;

   best_model = candidate_models{chosen(i)};
   model = add_model(model, best_model); %, 1); % 1 indicates adding a spatial model

   model.part(num_parts).spat_const = [0 1 0.8 1 0 1];
end


[model.part.computed] = deal(0);



result_str = fullfile(base_dir, sprintf('part_detections.mat'));
%load(result_str, 'cached_scores_test', 'cached_scores');

[model.part.computed] = deal(0);
[dk cached_scores] = collect_boost_data(model, D, cached_scores);
[dk cached_gt_pos] = collect_boost_data(model, Dpos, cached_gt_pos);


if(trainval==0) % For testing poselets
   cached_gt_test = get_gtbest_pos_reg(Dtest, cached_scores_test, cls);
   [Dpos_test pos_inds] = LMquery(Dtest, 'object.name', cls, 'exact');
   cached_gt_pos_test = cached_gt_test(pos_inds);
   [dk cached_gt_pos_test] = collect_boost_data(model, Dpos_test, cached_gt_pos_test);
else
   cached_gt_pos_test = [];
end

[dk cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);


save(result_str, 'model', 'cached_scores', 'cached_gt_pos', 'cached_scores_test', 'cached_gt_pos_test');
