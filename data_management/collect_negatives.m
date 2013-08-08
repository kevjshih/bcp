function [feats hyp] = mine_examples(model, D, cached_scores)
% Overview:
% For each image
% Load regions, assign labels: POS, NEG, Don't Care
% Load cached features (if they exist)
% Compute scores of each region (inference)

%model.thresh = -1;

dirs = [];
BDglobals;
%parfor_progress(length(D));
parfor ex = 1:length(D)
%    parfor_progress;
   [feats{ex} hyp{ex}] = mine_example(model, D(ex).annotation, cached_scores{ex}, dirs); 
end
%parfor_progress(0);

function [feats hyp] = mine_example(model, annotation, cached_scores, dirs)
   % To define: region_dir, feat_dir, im_dir
   cls = model.cls;
 
   [dk bn dk] = fileparts(annotation.filename);
   feat_file = fullfile(dirs.feat_dir, [bn '.mat']);
   label_dir = fullfile(dirs.label_dir, cls);
   label_file = fullfile(label_dir, [bn '.mat']);
   region_file = fullfile(dirs.region_dir, [bn '_proposals.mat']);

   regions = cached_scores.regions;
   labels = cached_scores.labels;

   if(isfield(model, 'incremental_feat') && model.incremental_feat==1)
      scores = cached_scores.incremental_feat;
   else
      scores = cached_scores.scores;
   end

   if(numel(labels)==0)
      feats = [];
      hyp = [];
      labels = [];
      return;
   end
   
   if(isfield(model, 'hard_local') && model.hard_local==1)
      negative = labels<0;
   else
      negative = labels==-1;
   end

   regions(~negative,:) = [];
   labels(~negative) = [];
   scores(~negative, :) = [];

   if(numel(labels)==0)
      feats = [];
      hyp = [];
      labels = [];
      return;
   end
   
   % TODO also need to load precomputed cached detections

   im_file = fullfile(dirs.im_dir, annotation.filename);
   % Find the hypothesis!
   if(isfield(model, 'loc_model') && model.loc_model==1)
      [hyp feat_data] = inference(im_file, model, find(negative), scores);
   else
      im = imread(im_file);
      [hyp feat_data] = inference(im, model, regions, scores);
   end

   if(isfield(model, 'nms'))
      nms_amount = model.nms;
   else
      nms_amount = 0.5;
   end
   
   if(isfield(model, 'weighted') && model.weighted==1)
      scores = [hyp.final_score];
      y = -1;
      weights = 1./(1+exp(y*[hyp.cached_score]));
      overall_scores = -weights.*(1-y*scores); % Negated since we want min, but nms finds max

      I = nms_iou([regions([hyp.region], :),  overall_scores(:)], nms_amount);
   else
      I = nms_iou([regions([hyp.region], :),  cat(1,hyp.final_score)], nms_amount);
   end
   hyp = hyp(I);
   % Compute features for each region
   feats = hyp_to_feat(model, hyp, feat_data);

   if(~isempty(feats))
      %feat_score = feats(1:end-1,:)'*model.part(end).filter(:) + model.part(end).bias + feats(end,:)'*model.cached_weight;
      w = get_model_weights(model);
      feat_score = w(1:end-1)'*feats + w(end);
      if(any(abs(feat_score - cat(2, hyp.final_score))>1e-9))
         fprintf('Some score differed by %f\n', max(abs(feat_score - cat(2, hyp.final_score))))
      end
   end
