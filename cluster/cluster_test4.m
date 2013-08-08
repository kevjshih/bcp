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
   load_init_final;
   load_init_test;
else % Just the training set
   set_str = 'train';
   C  = 15;
   load_init_data;
end

% Make sure the same parts are used for all cases, consistent is most constrainted, so all others should be supersets of it
consistent_base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, 'consistent'));

base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));

if(~exist(base_dir, 'file'))
   mkdir(base_dir);
end

cached_gt = get_gtbest_pos_reg(D, cached_scores, cls);
[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');

cached_gt_pos = cached_gt(pos_inds);

%%%%%%%%%%%%%%%% Do it! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



result_str = fullfile(base_dir, sprintf('part_detections.mat'));
if(~exist(result_str, 'file'))
num_parts = 0;

for i = 1:Ntodo
   fstr = fullfile(base_dir, sprintf('part_model_%d.mat', i));


   if(trainval)
      if(~exist(fstr, 'file'))
         fprintf('Part %d (%s) wasn''t found!\n', i, fstr);
         continue;
      end
   else
      consistent_fstr = fullfile(consistent_base_dir, sprintf('part_model_%d.mat', i));
      if(~exist(consistent_fstr, 'file'))
         fprintf('Part %d (%s) wasn''t found for consistent model!\n', i, consistent_fstr);
         continue;
      end

      if(~exist(fstr, 'file'))
         error('Part %d (%s) is missing but it worked for the consistent model!\n', i, fstr);
      end
   end

   num_parts = num_parts + 1;

   cur_model = load(fstr, 'model', 'w_all');      
  
   if(i==1)
      model = cur_model.model;
   else
      model.part(num_parts) = orderfields(cur_model.model.part, model.part); % For some reason the structures don't line up....
      model.num_parts = num_parts;
   end

   model.part(num_parts).computed = 0;

%   [dk cached_scores] = collect_boost_data_loo(model, D, cached_scores, cur_model.w_all);
%   [dk cached_gt_pos] = collect_boost_data_loo(model, Dpos, cached_gt_pos, cur_model.w_all);

%   model.part(num_parts).computed = 1;
end

[model.part.computed] = deal(0);
%[dk cached_scores] = collect_boost_data(model, D, cached_scores);%, cur_model.w_all);
[dk cached_gt_pos] = collect_boost_data(model, Dpos, cached_gt_pos);

[dk cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);


else
   load(result_str);
end


[model.part.computed] = deal(0);

if(~exist('cached_gt_pos_test', 'var') || trainval==0) % For testing poselets
   cached_gt_test = get_gtbest_pos_reg(Dtest, cached_scores_test, cls);
   [Dpos_test pos_inds] = LMquery(Dtest, 'object.name', cls, 'exact');
   cached_gt_pos_test = cached_gt_test(pos_inds);
   [dk cached_gt_pos_test] = collect_boost_data(model, Dpos_test, cached_gt_pos_test);
else
   cached_gt_pos_test = [];
end

[model.part.computed] = deal(1);

%result_str = fullfile(base_dir, sprintf('part_detections.mat'));

save(result_str, '-v7.3', 'model', 'cached_gt_pos', 'cached_scores_test', 'cached_gt_pos_test');
