function new_model = train_region_model(D, cached_scores, model)

BDglobals;

model_file = fullfile(WORKDIR, [model.cls '_region_local_model.mat']);
region_feat_dir = fullfile(dirs.feat_dir, 'region');

% Check to see if it's already computed
if(exist(model_file, 'file'));
   load(model_file, 'region_local_model');
   new_model.region_local_model = region_local_model;
   return
end


% Load data
for i = 1:length(D)
   if(any(cached_scores{i}.labels==1)) % positive image
      fprintf('%d/%d\n', i, length(D));
      [dk bn dk] = fileparts(D(i).annotation.filename); 
      region_file = fullfile(region_feat_dir, [bn '_regfeat.mat']);
  
      if(~exist(region_file, 'file')) 
         continue;
      end
      
   
      feats = load(region_file, 'c_hist', 't_hist', 'region_hog');
      c_hist = cat(2, feats.c_hist{:});
      t_hist = cat(2, feats.t_hist{:});
     
      reg_hog = cell(1, length(feats.region_hog)); 
      for j = 1:length(feats.region_hog)
         reg_hog{j} = feats.region_hog{j}(:);
      end
      reg_hog = cat(2, reg_hog{:});
   
      annotation = D(i).annotation;
      cls = model.cls;
      regions = cached_scores{i}.regions;
      % Find best regions
      boxes = LMobjectboundingbox(annotation, cls);
      all_overlaps = bbox_overlap_mex(boxes, regions);
      [overlaps best_ind] = max(all_overlaps, [], 2);
 
      best_inds = best_ind(overlaps>=0.5);

      positives{i} = [c_hist(:,best_inds); t_hist(:,best_inds); reg_hog(:,best_inds)];

      overlaps = max(all_overlaps, [], 1);
      best_inds = find(overlaps<0.35 & overlaps>0.1);
      negatives{i} = [c_hist(:,best_inds); t_hist(:,best_inds); reg_hog(:,best_inds)];
   end
end


positives_tr = cat(2, positives{:});
negatives_tr = cat(2, negatives{:});
labels_tr = [ones(1, size(positives_tr,2)), -ones(1, size(negatives_tr,2))];
examples_tr = [positives_tr, negatives_tr];

%w0 = svm_dual_mex(labels, examples, 1e-1);
[w0{1} b] = fast_svm(labels_tr, examples_tr(1:128,:), 1e-1, 10); % This one doesn't really work
[w0{2} b] = fast_svm(labels_tr, examples_tr(128 + [1:256],:), 1e-5, 10); % This one doesn't really work either
[w0{3} b] = fast_svm(labels_tr, examples_tr([128+256+1:end],:), 1e-3, 20);

region_local_model = w0;
save(model_file, 'region_local_model');
new_model.region_local_model = region_local_model;
return;

