function [model w_loo all_models] = train_boosted(model, D, cached_scores, num_iter, Nouter, rate, weighted_subset)

if(~exist('weighted_subset','var'))
   weighted_subset = 0;
end

if(~exist('num_iter', 'var'))
    num_iter = Inf;
end
    

if(~exist('Nouter', 'var'))
   Nouter = 10;
end

if(~exist('rate', 'var'))
   rate = 1;
end

blocksize = 300;
cls = model.cls;
% Setup data
[dk inds] = LMquery(D, 'object.name', cls, 'exact');
Dpos = D(inds);
cached_pos = cached_scores(inds);

Dneg = D;
Dneg(inds) = [];
cached_neg = cached_scores;
cached_neg(inds) = [];

N = length(Dneg);
%Nouter = 10;

% Count number of negative windows
Nneg = 0;
for i = 1:length(cached_neg)
   Nneg = Nneg + length(cached_neg{i}.labels);
end

neg_feats = cell(N,1);

all_models = {};

model.cached_weight = 0; % Don't use previous score as a feature
model.weighted = 1;

for outer_iter = 1:Nouter

   [pos_feats0 pos_hyp] = collect_positives(model, Dpos, cached_pos);
   pos_feats = cat(2, pos_feats0{:});
   r = randperm(N);

   for inner_iter = 1:min(num_iter, ceil(N/blocksize))
      first = (inner_iter-1)*blocksize+1;
      last = min(inner_iter*blocksize, N);
      block = sort(r(first:last));

      neg_feats(block) = collect_negatives(model, Dneg(block), cached_neg(block));
     
      % Update model
      feats = [pos_feats, cat(2, neg_feats{:})];
      log_weights = feats(end, :);
      feats = feats(1:end-1, :);

      labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];
      weights = 1./(1+exp(labels.*log_weights'));

      Z_pos = sum(weights(labels==1));
      Z_neg = sum(weights(labels==-1))*Nneg/((sum(labels==-1))+eps); % Estimate of all negative mass

      weights_norm = weights;
      weights_norm(labels==1) = weights_norm(labels==1)/Z_pos;
      weights_norm(labels==-1) = weights_norm(labels==-1)/Z_neg;

      fprintf('Iter: %d/%d  - %d/%d\n', outer_iter, Nouter, inner_iter, ceil(N/blocksize));
      reg = ones(size(feats,1)+1,1);
      reg(end-1:end) = 0.01;

      if(weighted_subset)
         [w deltas alphas] = weighted_subset_svm(feats, labels, weights_norm, 100, rate);
      else
         [w deltas alphas] = weighted_subset_svm(feats, labels, weights_norm, 100, ceil(sum(labels==1)*rate));
      end

      clear feats; 
      all_models{end+1} = model;
      w_update = w;
      w_update(end+1) = w(end);
      w_update(end-1) = 0; % 

      model = update_model_weights(model, w_update);
   end
end

% Compute which image each positive feature came from
for i = 1:length(pos_hyp)
   pos_im_ind{i} = repmat(i, length(pos_hyp{i}),1);
end

pos_im_inds = cat(1, pos_im_ind{:});

% Given final deltas, recompute w for examples that were included
delta_pos = deltas(1:length(pos_im_inds));
delta_neg = deltas(length(pos_im_inds)+1:end);

%alpha_pos = alphas(1:length(pos_im_inds));
%alpha_neg = alphas(length(pos_im_inds)+1:end);

pos_to_do = find(delta_pos);

pos_im_todo = unique(pos_im_inds(pos_to_do));

fast_subset=1;
for i = 1:length(pos_im_todo)
   curim = pos_im_todo(i);

   delta1 = [delta_pos(pos_im_inds~=curim); delta_neg];
   %alpha0 = [alpha_pos(pos_im_inds~=curim); alpha_neg];
   pos_feats = cat(2, pos_feats0{[1:curim-1, curim+1:end]});
   feats = [pos_feats, cat(2, neg_feats{:})];
   log_weights = feats(end, :);
   feats = feats(1:end-1, :);

   labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];
   weights = 1./(1+exp(labels.*log_weights'));

   Z_pos = sum(weights(labels==1));
   Z_neg = sum(weights(labels==-1))*Nneg/((sum(labels==-1))+eps); % Estimate of all negative mass

   weights_norm = weights;
   weights_norm(labels==1) = weights_norm(labels==1)/Z_pos;
   weights_norm(labels==-1) = weights_norm(labels==-1)/Z_neg;

   if(fast_subset) % Just train on precomputed subset
      [w_loo_t alphas_loo{curim}] = svm_weighted_dual_mex(labels(delta1==1), feats(:, delta1==1), weights_norm(delta1==1), 100, []);%, alpha0);
   elseif(weighted_subset)
      [w_loo_t deltas_loo{curim} alphas_loo{curim}] = weighted_subset_svm(feats, labels, weights, 1e-1, rate);
   else
      [w_loo_t deltas_loo{curim} alphas_loo{curim}] = weighted_subset_svm(feats, labels, weights, 5e-2, ceil(sum(labels==1)*rate));
   end
   w_update = w_loo_t;
   w_update(end+1) = w_loo_t(end);
   w_update(end-1) = 0; % 

   w_loo{curim} = w_update;
end

for i = 1:length(pos_feats0)
   if(~ismember(i, pos_im_todo))
      w_loo{i} = w_update;
   end
end

return
if(0)
% Compare deltas
real_deltas = -ones(length(pos_im_inds), length(pos_im_todo));

for i = 1:length(pos_im_todo)
   curim = pos_im_todo(i);
   notin = find(pos_im_inds~=curim);

   real_deltas(notin, i) = deltas_loo{curim}(1:length(notin));  
end

   if(~isempty(pos_feats0{i}));
      scores_loo{i} = w_loo{i}(1:end-1)'*pos_feats0{i} + w_loo{i}(end);
      scores_orig{i} = w(1:end-1)'*pos_feats0{i} + w(end);
   end
end
end
