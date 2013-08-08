function cluster_test(cls)

% Need the latest version
addpath external/voc-release4
% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
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

set_str = 'train';
load_init_data;

% Make sure the same parts are used for all cases, consistent is most constrainted, so all others should be supersets of it

base_dir = fullfile('data/results', cls, 'part_models_train_dpm');

if(~exist(base_dir, 'file'))
   mkdir(base_dir);
end

load(sprintf('~kjshih2/boosted_detection/external/voc-release4/dpm_data/voc_cache_v4/2010/900_ex_3_comp_%s/1/%s_final_900_ex_3_comp_%s.mat', cls,cls,cls));
model = prepare_dpm(model);



cached_gt = get_gtbest_pos_reg(D, cached_scores, cls);
[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');

cached_gt_pos = cached_gt(pos_inds);
%%%%%%%%%%%%%%%% Do it! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[dk cached_scores] = collect_boost_data(model, D, cached_scores);
[dk cached_gt_pos] = collect_boost_data(model, Dpos, cached_gt_pos);




cached_gt_test = get_gtbest_pos_reg(Dtest, cached_scores_test, cls);
[Dpos_test pos_inds] = LMquery(Dtest, 'object.name', cls, 'exact');
cached_gt_pos_test = cached_gt_test(pos_inds);
[dk cached_gt_pos_test] = collect_boost_data(model, Dpos_test, cached_gt_pos_test);

%[dk cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test);

result_str = fullfile(base_dir, sprintf('part_detections.mat'));

%save(result_str, 'model', 'cached_scores', 'cached_gt_pos', 'cached_scores_test', 'cached_gt_pos_test');
save(result_str, 'model', 'cached_gt_pos_test', 'cached_gt_pos');
