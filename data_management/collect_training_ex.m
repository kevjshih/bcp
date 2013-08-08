function [feats hyp] = collect_training_ex(model, D, cached_scores, label)
% Combines collect_{positives,negatives}
% Overview:
% For each image
% Load regions, assign labels: POS, NEG, Don't Care
% Load cached features (if they exist)
% Compute scores of each region (inference)

if(label==1)
   model.thresh = -inf; % Don't want to prune potential positives
end

dirs = [];
BDglobals;

parfor ex = 1:length(D)
   if(label==1) % To preserve old behavior
   %   fprintf('%d/%d\n', ex, length(D));
   end
   [feats{ex} hyp{ex}] = mine_example(model, D(ex).annotation, cached_scores{ex}, dirs, label); 
end


function [feats hyp] = mine_example(model, annotation, cached_scores, dirs, label)
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

   if(label==1)
      has_label = labels>0;
   elseif(label==-1)
      if(isfield(model, 'hard_local') && model.hard_local==1)
         has_label = labels<=0;
      else
         has_label = labels<0;
      end   
   else
      error('Label should be -1 or 1!\n');
   end

   regions(~has_label,:) = [];
   labels(~has_label) = [];
   scores(~has_label, :) = [];

   if(~any(has_label)) % This is a negative example, or no good proposals
      feats = [];
      hyp = [];
      labels = [];
      return;
   end

   im_file = fullfile(dirs.im_dir, annotation.filename);

   if(isfield(model, 'loc_model') && model.loc_model==1)
      [hyp feat_data] = inference(im_file, model, find(has_label), scores);
   else

      if(isfield(model, 'bound_feat') && model.bound_feat==1)
         im = get_bound_feat(model, annotation.filename, dirs);
      else 
         im = imread(im_file);
      end

      [hyp feat_data] = inference(im, model, regions, scores);
   end

   if(isfield(model, 'nms'))
      nms_amount = model.nms;
   else
      nms_amount = 0.3;
   end

   % Not worrying about weighted scores here, it's not clear how to actually handle this
   if(label==1)
%      if(isfield(model, 'weighted') && model.weighted==1)
%         hyp = get_best_hyps_weighted(hyp, labels);
%      else
         hyp = get_best_hyps(hyp, labels);
%      end
   
      hyp = prune_hyp(hyp);
   else
      if(isfield(model, 'weighted') && model.weighted==1)
         scores = [hyp.final_score];
         y = -1;
         weights = 1./(1+exp(y*[hyp.cached_score])); % Find examples that will contribute most to the loss
         overall_scores = weights.*log(1+exp(-y*scores)); 
  
         I = nms_iou([regions([hyp.region], :),  overall_scores(:)], nms_amount);
      else
         I = nms_iou([regions([hyp.region], :),  cat(1,hyp.final_score)], nms_amount);
      end
      hyp = hyp(I);
   end

   % Compute features for each region
   feats = hyp_to_feat(model, hyp, feat_data);

   if(~isempty(feats))
      %feat_score = feats(1:end-1,:)'*model.part(end).filter(:) + model.part(end).bias + feats(end,:)'*model.cached_weight;
      w = get_model_weights(model);
      feat_score = w(1:end-1)'*feats + w(end);
      if(any(abs(feat_score - cat(2, hyp.final_score))>1e-7))
         fprintf('Some score differed by %f\n', max(abs(feat_score - cat(2, hyp.final_score))));
      end
   end

function hyp_out = get_best_hyps(hyp, labels)

   true_label = labels([hyp.region]); % Gets the object ind
   scores = [hyp.final_score];
   
   [ind dk un_label] =  unique(true_label);

   for i = 1:length(ind)
      this_ind = find(un_label==i);
      [best_score best_ind] = max(scores(this_ind));
      hyp_out(i) = hyp(this_ind(best_ind));
   end

function hyp_out = get_best_hyps_weighted(hyp, labels)
   % Find the hypothesis that minimizes the weighted svm loss for each example

   true_label = labels([hyp.region]); % Gets the object ind
   scores = [hyp.final_score];
   weights = 1./(1+exp([hyp.cached_score]));

   [ind dk un_label] =  unique(true_label);

   for i = 1:length(ind)
      this_ind = find(un_label==i);
      [best_score best_ind] = min(weights(this_ind).*(-log(1+exp(scores(this_ind)))));
      hyp_out(i) = hyp(this_ind(best_ind));
   end

function hyp = prune_hyp(hyp)
% Remove examples that weren't assigned a valid part
% We don't care about these during learning

scores = [hyp.final_score];

hyp(isinf(scores)) = [];



