function cluster_test(cls, trainval, refinement_type)

% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
end

try
   trainval = str2num(trainval);
end

%matlabpool_robust;

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
else % Just the training set
   set_str = 'train';
end

base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
result_str = fullfile(base_dir, sprintf('part_evaluation.mat'));

if(~exist(result_str, 'file'))
   if(trainval)
      set_str = 'trainval';
      load_init_final;
      load_init_test;
   else % Just the training set
      set_str = 'train';
      C  = 15;
      load_init_data;
   end


   det_str = fullfile(base_dir, sprintf('part_detections.mat'));
   load(det_str, 'model', 'cached_gt_pos_test', 'cached_scores_test');

   kp_str = fullfile(base_dir, sprintf('kp_results.mat'));
   load(kp_str, 'model', 'per_kp_med', 'keypoint_name');

   mean_kp_ap = mean(mean(cat(2, per_kp_med{:})));
   max_kp_ap = (max(cat(2, per_kp_med{:}), [], 2));
   fprintf('|');
   for i = 2:model.num_parts-1
      fprintf('=')
   end
   fprintf('|\n');

   for i = 1:model.num_parts
      fprintf('.')
      [recall{i} prec{i} ap(i) pos_prec{i}] = test_part_detections_D(cls, Dtest, cached_scores_test, i);
   end   
   fprintf('\n');

   pos_prec_mat = [];
   for i = 1:length(pos_prec)
      pos_prec_mat = [pos_prec_mat, cat(2, pos_prec{i}{:})'];
   end

   max_ap = max(pos_prec_mat, [], 2);
   max_ap_cum = cummax(pos_prec_mat, 2);


   save(result_str, 'mean_kp_ap', 'max_kp_ap', 'max_ap', 'max_ap_cum', 'recall', 'prec', 'ap', 'pos_prec', 'pos_prec_mat')
else % if(~exist(result_str, 'file'))
   load(result_str);
end

% To report:
% APM
kp_str = fullfile(base_dir, sprintf('kp_results.mat'));
load(kp_str, 'per_kp_med');

for i = 1:length(per_kp_med)
   [a b] = sort(per_kp_med{i},'descend');
   m_top3(i) = mean(a(1:3));
end

fprintf('%s:%s (%s)\n', cls, refinement_type, set_str);
%fprintf(' mAP\t AMP\t mKP\t max KP\t mTop3\n');
%fprintf('&%.1f &\t %.1f&\t %.1f&\t %.1f\t %.1f\n', 100*mean(ap), 100*mean(max_ap), 100*mean_kp_ap, 100*mean(max_kp_ap), 100*mean(m_top3));

fprintf('&%.1f\t %.1f\t &%.1f\n', 100*mean(ap), 100*mean(m_top3), 100*mean(max_kp_ap));
