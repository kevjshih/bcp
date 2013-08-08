function [feats hyp] = mine_examples(model, D, cached_scores)
% Overview:
% For each image
% Load regions, assign labels: POS, NEG, Don't Care
% Load cached features (if they exist)
% Compute scores of each region (inference)

model.thresh = -inf; % Don't want to prune potential positives

dirs = [];
BDglobals;
parfor ex = 1:length(D)
   fprintf('%d/%d\n', ex, length(D));
   [feats{ex} hyp{ex}] = mine_example(model, D(ex).annotation, cached_scores{ex}, dirs); 
end

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

   positive = labels>0;
   regions(~positive,:) = [];
   labels(~positive) = [];
   scores(~positive, :) = [];

   if(~any(positive)) % This is a negative example, or no good proposals
      feats = [];
      hyp = [];
      return;
   end   

   im_file = fullfile(dirs.im_dir, annotation.filename);

   if(isfield(model, 'loc_model') && model.loc_model==1)
      [hyp feat_data] = inference(im_file, model, find(positive), scores);
   else
      im = imread(im_file);
      [hyp feat_data] = inference(im, model, regions, scores);
   end
   %save(feat_file, 'feat_data', '-v6');

   if(isfield(model, 'weighted') && model.weighted==1)
      hyp = get_best_hyps_weighted(hyp, labels);
   else
      hyp = get_best_hyps(hyp, labels);
   end

   hyp = prune_hyp(hyp);

   % Compute features for each region
   feats = hyp_to_feat(model, hyp, feat_data);

   if(isempty(feats))
      return;
   end

   if(0)
   feat_score = feats(1:end-1,:)'*model.part(end).filter(:) + model.part(end).bias + feats(end,:)'*model.cached_weight;

   if(any(abs(feat_score - cat(1, hyp.final_score))>1e-9))
      fprintf('Some score differed by %f\n', max(abs(feat_score - cat(1, hyp.final_score))))
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
      [best_score best_ind] = min(weights(this_ind).*(1-scores(this_ind)));
      hyp_out(i) = hyp(this_ind(best_ind));
   end

function hyp = prune_hyp(hyp)
% Remove examples that weren't assigned a valid part
% We don't care about these during learning

scores = [hyp.final_score];

hyp(isinf(scores)) = [];



