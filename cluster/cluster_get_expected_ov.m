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

[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');
[model.part.computed] = deal(1);

det_str = fullfile(base_dir, sprintf('part_detections.mat'));
load(det_str);


for i = 1:model.num_parts
   fprintf('%d\n', i);
   [model.part.computed] = deal(1);
   model.part(i).computed = 0;

   [reference_box{i}] = get_expected_pos(model, Dpos, cached_gt_pos, 0);
end

for i = 1:length(cached_scores)
   overlaps{i} = zeros(size(cached_scores{i}.regions,1), model.num_parts);
   calib_scores{i} = zeros(size(cached_scores{i}.regions,1), model.num_parts);
end

for i = 1:length(cached_scores_test)
   overlaps_test{i} = zeros(size(cached_scores_test{i}.regions,1), model.num_parts);
   calib_scores_test{i} = zeros(size(cached_scores_test{i}.regions,1), model.num_parts);
end


loc_str = fullfile(base_dir, sprintf('loc_model.mat'));
load(loc_str, 'model_wbox');

for i = 1:model.num_parts
   fprintf('%d\n', i);

         inds = [1:4] + 4*(i-1);
   for j = 1:length(cached_scores)
      if(~isempty(cached_scores{j}.regions))
         expected_box = predict_expected_pos(cached_scores{j}.regions, reference_box{i}, cached_scores{j}.part_trans(:, i)==2);
   
%         for k = 1:size(expected_box,1)
            overlaps{j}(:, i) = bbox_overlap_pw(expected_box, double(cached_scores{j}.part_boxes(: ,inds)));
            %overlaps{j}(k, i) = diag(bbox_overlap_mex(expected_box(k, :), double(cached_scores{j}.part_boxes(k, inds)));
%         end
            calib_scores{j}(:, i) =  sigmoid(cached_scores{j}.part_scores(:, i), model_wbox.part(i).calib);
      end
   end


   for j = 1:length(cached_scores_test)
      if(~isempty(cached_scores_test{j}.regions))
         inds = [1:4] + 4*(i-1);
         expected_box = predict_expected_pos(cached_scores_test{j}.regions, reference_box{i}, cached_scores_test{j}.part_trans(:, i)==2);
   
         overlaps_test{j}(:, i) = bbox_overlap_pw(expected_box, double(cached_scores_test{j}.part_boxes(: ,inds)));
         calib_scores_test{j}(:, i) =  sigmoid(cached_scores_test{j}.part_scores(:, i), model_wbox.part(i).calib);
      end
   end
end

res_str = fullfile(base_dir, sprintf('part_expected_overlaps.mat'));
save(res_str, 'overlaps', 'overlaps_test', 'calib_scores*', 'reference_box');

