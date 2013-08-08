function cluster_localization(cls, trainval, refinement_type)

% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
end

try
   trainval = str2num(trainval);
end

%matlabpool_robust;

%%%%%%%%%%%%%%%%% Setup class... %%%%%%%%%%%%%%%%%%%%%%%%%%
startup_cluster;
get_cluster_basedir;
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
   load_init_data;
end

base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
det_str = fullfile(base_dir, sprintf('part_detections.mat'));
load(det_str, 'cached_scores', 'cached_scores_test', 'cached_gt_pos');

%result_str = fullfile(base_dir, sprintf('boost_model.mat'));
%load(result_str, 'model');

%%%%%%%%%%%%%%%%%%%% Do It! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loc_str = fullfile(base_dir, sprintf('loc_model.mat'));
load(loc_str, 'model_wbox');

cached_scores_box = box_inference(cached_scores, model_wbox);
cached_scores_box = apply_weak_learner(cached_scores_box, model_wbox.learner);

cached_scores_test_box = box_inference(cached_scores_test, model_wbox);
cached_scores_test_box = apply_weak_learner(cached_scores_test_box, model_wbox.learner);

clear cached_scores, cached_scores_test


[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');
% Now get reference boxes.....
for i = 1:model_wbox.num_parts
   fprintf('%d\n', i);
   [model_wbox.part.computed] = deal(1);
   model_wbox.part(i).computed = 0;

   [reference_box{i}] = get_expected_pos(model_wbox, Dpos, cached_gt_pos, 0);
end


for i = 1:length(cached_scores_box)
   cached_scores_box{i}.overlaps = zeros(size(cached_scores_box{i}.regions,1), model_wbox.num_parts);
   cached_scores_box{i}.calib_scores{i} = zeros(size(cached_scores_box{i}.regions,1), model_wbox.num_parts);
end

for i = 1:length(cached_scores_test_box)
   cached_scores_test_box{i}.overlaps = zeros(size(cached_scores_test_box{i}.regions,1), model_wbox.num_parts);
   cached_scores_test_box{i}.calib_scores{i} = zeros(size(cached_scores_test_box{i}.regions,1), model_wbox.num_parts);
end



for i = 1:model_wbox.num_parts
   fprintf('%d\n', i);

         inds = [1:4] + 4*(i-1);
   for j = 1:length(cached_scores_box)
      if(~isempty(cached_scores_box{j}.regions))
         expected_box = predict_expected_pos(cached_scores_box{j}.regions, reference_box{i}, cached_scores_box{j}.part_trans(:, i)==2);
   
%         for k = 1:size(expected_box,1)
         cached_scores_box{j}.overlaps(:, i) = bbox_overlap_pw(expected_box, double(cached_scores_box{j}.part_boxes(: ,inds)));
            %overlaps{j}(k, i) = diag(bbox_overlap_mex(expected_box(k, :), double(cached_scores{j}.part_boxes(k, inds)));
%         end
         cached_scores_box{j}.calib_scores(:, i) =  sigmoid(cached_scores_box{j}.part_scores(:, i), model_wbox_wbox.part(i).calib);
      end
   end


   for j = 1:length(cached_scores_test_box)
      if(~isempty(cached_scores_test_box{j}.regions))
         inds = [1:4] + 4*(i-1);
         expected_box = predict_expected_pos(cached_scores_test_box{j}.regions, reference_box{i}, cached_scores_test_box{j}.part_trans(:, i)==2);
   
         cached_scores_test_box{j}.overlaps(:, i) = bbox_overlap_pw(expected_box, double(cached_scores_test_box{j}.part_boxes(: ,inds)));
         cached_scores_test_box{j}.calib_scores(:, i) =  sigmoid(cached_scores_test_box{j}.part_scores(:, i), model_wbox_wbox.part(i).calib);
      end
   end
end






result_str = fullfile(base_dir, sprintf('loc_results.mat'));
save(result_str, '-v7.3', 'cached_scores_box', 'cached_scores_test_box', 'reference_box');
