function cluster_refinement(cls, trainval, refinement_type, block_size, block_ind)

startup_cluster

% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
end

try, trainval = str2num(trainval); end
try, block_size = str2num(block_size); end
try, block_ind = str2num(block_ind); end

matlabpool_robust;

%%%%%%%%%%%%%%%%% Setup class... %%%%%%%%%%%%%%%%%%%%%%%%%%
BDglobals;

BDpascal_init;
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
   candidate_file = fullfile('data', [cls '_candidates_trainval.mat']);

   load_init_data;
   clear Dtest cached_scores_test;
end

load(candidate_file);

%%%%%%%%%%%%%%%%% Set up model flags %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.hard_local = 0;
model.score_feat = 0;
model.weighted = 0;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20];
model.cached_weight = 0;


%%%%%%%%%%%%%%%% Set up other stuff.... %%%%%%%%%%%%%%%%%%%%%%%%
switch refinement_type
   case 'new_baseline'
      model.min_ov = -inf;
      model.min_prob = 0.3;
   case 'old_baseline'
      model.min_ov = -inf;
      model.min_prob = -inf;
   case 'consistent'
      model.min_ov = 0.75;
      model.min_prob = 0.2;
end
empty_model = model;

base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));

if(~exist(base_dir, 'file'))
   mkdir(base_dir);
end

cached_gt = get_gt_pos_reg(D, cached_scores, cls);

%%%%%%%%%%%%%%%% Train it! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
todo = (block_ind-1)*block_size + [1:block_size];
fprintf('Doing blocks %d to %d\n', todo(1), todo(end));

for i = todo
   try

   fstr = fullfile(base_dir, sprintf('part_model_%d.mat', i));

   if(exist(fstr, 'file'))
      fprintf('Part %d (%s) is already done\n', i, fstr);
   end

   best_model = candidate_models{chosen(i)};

   model = add_model(empty_model, best_model); %, 1); % 1 indicates adding a spatial model

   model.part.bias = model.part.bias + 0.5; % Make sure to pull in plenty of hard negatives
   model.part.spat_const = [0 1 0.8 1 0 1];

   switch refinement_type
      case {'new_baseline', 'consistent'}
         [model w_all] = train_consistency(model, D, cached_gt, C);
      case 'old_baseline'
%         [model neg_feats] = train_loo_cache(model, D, cached_gt, 10, 2, 0.5, C);
         [model neg_feats w_all w_noloo all_models] = train_loo_cache(model, D, cached_gt, 10, 5, 1, C, neg_feats);
       case 'poselet'
         [model neg_feats w_all] = train_fixed_poselet(model, D, cached_gt, 10, 2, 0.50, 0.3, C); 
       case 'poselet_loc'
         [model neg_feats w_all] = train_loc_poselet(model, D, cached_gt, 10, 5, 0.10, 0.3, C); 
   end


   save(fstr, 'model', 'w_all');      

   catch
      fprintf('Error processing %s\n', fstr);
   end
end
