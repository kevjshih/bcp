function val_refinement_final(cls, cls_loo, trainval, IGNORE_SIMILAR)
%javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');



% Parse input
try, trainval = str2num(trainval); end
try, IGNORE_SIMILAR = str2num(IGNORE_SIMILAR); end
try, if(~isinf(str2num(cls))), cls=str2num(cls); end, end
try, if(~isinf(str2num(cls_loo))), cls_loo=str2num(cls_loo); end, end


%%% Setup class...
BDglobals;

BDpascal_init;
VOCopts.sbin = 8;
VOCopts.localdir = fullfile(WORKDIR, 'exemplars');

if(isnumeric(cls))
   clsind = cls;
   cls = VOCopts.classes{clsind};
end

if(isnumeric(cls_loo))
   cls_looind = cls_loo;
   cls_loo = VOCopts.classes{cls_looind};
end

if(strcmp(cls, cls_loo))
   fprintf('You can''t leave out the positive category!!\n');
   return;
end

startup_cluster
matlabpool_robust;

fprintf('Doing category: %s - LOO %s\n', cls, cls_loo);
%%%%%%%%%%%%%%%%%%%%%%%%%

if(~exist('IGNORE_SIMILAR', 'var'))
   IGNORE_SIMILAR = 0;
end

if(~exist('trainval', 'var'))
   trainval = 0;
end

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


%%%%%%%%%%%%%% Resume
fname0 = fullfile(base_dir, 'part_detections.mat');
fname = fullfile(base_dir, sprintf('part_detections_loo_%s.mat', cls_loo));

orig_model = load(fname0, 'model');

%if(exist(fname, 'file'))
try
   load(fname, 'model');
   fprintf('Resuming!\n');
   fprintf('%d parts already finished\n', model.num_parts);
catch
   model = empty_model;
end

%%%%%%%%%%%%% Select parts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

% Remove everything from cls_loo
[Dc looind] = LMquery(D, 'object.name', cls_loo);
D(looind) = [];
cached_scores(looind) = [];

cached_gt = get_gt_pos_reg(D, cached_scores, cls);

if(model.num_parts>0)
    [model.part.computed] = deal(1);
end

for i = 1:40
   iter = i;

   if(model.num_parts>=iter)
      fprintf('%d already computed!\n', i);
      continue;
   end

   if(iter==1)
      model.part = orig_model.model.part(i); % Special case to initialize structure fields
   else
      model.part(iter) = orig_model.model.part(i);
   end

   model.part(iter).bias = model.part(iter).bias + 1; % Want to bring in negatives
   model.num_parts = iter;

   model.part(i).computed = 0;
   try
      [model] = train_consistency(model, D, cached_gt, C, 1); % 1 iteration of train consistency
   catch
      w_all = []; % Model failed to train!
   end
   % Compute Part Scores
%   w_all_storage{iter} = w_all;
   model.part(iter).computed = 1;

   save(fname, '-v7.3', 'model');
end
