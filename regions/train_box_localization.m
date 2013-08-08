function new_model = train_region_model(D, cached_scores, model)

BDglobals;

model_file = fullfile(WORKDIR, [model.cls '_box_local_model.mat']);
region_feat_dir = fullfile(dirs.feat_dir, 'box');
new_model = model;

% Check to see if it's already computed
if(exist(model_file, 'file'));
   load(model_file, 'box_local_model');
   new_model.box_local_model = box_local_model;
   return;
end

% Load data
for i = 1:length(D)
   if(any(cached_scores{i}.labels==1)) % positive image
      fprintf('%d/%d\n', i, length(D));
      [dk bn dk] = fileparts(D(i).annotation.filename); 
      region_file = fullfile(region_feat_dir, [bn '_boxfeat.mat']);
  
      if(~exist(region_file, 'file')) 
         continue;
      end
         
      feats = load(region_file, 'box_feat');
      box_hog = cell(1, length(feats.box_feat)); 

      for j = 1:length(box_hog)
         box_hog{j} = feats.box_feat{j}(:);
      end
      box_hog = cat(2, box_hog{:});
   
      annotation = D(i).annotation;
      cls = model.cls;
      regions = cached_scores{i}.regions;
      % Find best regions
      boxes = LMobjectboundingbox(annotation, cls);
      all_overlaps = bbox_overlap_mex(boxes, regions);
      [overlaps best_ind] = max(all_overlaps, [], 2);
 
      best_inds = best_ind(overlaps>=0.5);

      positives{i} = [box_hog(:,best_inds)];

      overlaps = max(all_overlaps, [], 1);
      best_inds = find(overlaps<0.35 & overlaps>0.1);;
      negatives_loc{i} = [box_hog(:,best_inds)];
      best_inds = find(overlaps<0.1);
      negatives_bg{i} = [box_hog(:,best_inds)];
   end
end


positives_tr = cat(2, positives{:});
negatives_tr = cat(2, negatives_loc{:});
labels_tr = [ones(1, size(positives_tr,2)), -ones(1, size(negatives_tr,2))];
examples_tr = [positives_tr, negatives_tr];

keyboard
%w0 = svm_dual_mex(labels, examples, 1e-1);
[w0 b] = fast_svm(labels_tr, examples_tr, 1e-3, 20);

box_local_model = w0;
save(model_file, 'box_local_model');
new_model.box_local_model = box_local_model;
return;
