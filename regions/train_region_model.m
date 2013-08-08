function new_model = train_region_model(D, cached_scores, model)

BDglobals;

new_model = model;
%model_file = fullfile(WORKDIR, [model.cls '_region_model.mat']);
region_feat_dir = fullfile(dirs.feat_dir, 'region');

% Check to see if it's already computed
%if(exist(model_file, 'file'));
%   load(model_file, 'region_model');
%   new_model.region_model = region_model;
%   return
%end


% Load data
for i = 1:length(D)
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
   if(any(cached_scores{i}.labels>0)) % positive image
      boxes = LMobjectboundingbox(annotation, cls);
      [overlaps best_ind] = max(bbox_overlap_mex(boxes, regions), [], 2);
 
      best_inds = best_ind(overlaps>=0.5);

      positives{i} = [c_hist(:,best_inds); t_hist(:,best_inds); reg_hog(:,best_inds)];
   else % Negative image
      r = randperm(length(cached_scores{i}.labels));
      best_inds = r(1:min(end,10));
      negatives{i} = [c_hist(:,best_inds); t_hist(:,best_inds); reg_hog(:,best_inds)];
   end 
end

positives = cat(2, positives{:});
negatives = cat(2, negatives{:});

labels = [ones(1, size(positives,2)), -ones(1, size(negatives,2))];
examples = [positives, negatives];

clear negatives 

%w0 = svm_dual_mex(labels, examples, 1e-1);
[w0{1} b] = fast_svm(labels, examples(1:128,:), 1e-1, 10);
[w0{2} b] = fast_svm(labels, examples(128 + [1:256],:), 1e-1, 10);
[w0{3} b] = fast_svm(labels, examples([128+256+1:end],:), 1e-1, 10);

region_model = w0;
%save(model_file, 'region_model');
new_model.region_model = region_model;
return;
