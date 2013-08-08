function [model ] = train_boosted(model, D, cached_scores, num_iter, Nouter, rate)

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

neg_feats = cell(N,1);

all_models = {};

model.cached_weight = 0; % We're not using this here

for outer_iter = 1:Nouter

   [pos_feats pos_hyp] = collect_positives(model, Dpos, cached_pos);
   pos_feats = cat(2, pos_feats{:});
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

      fprintf('Iter: %d/%d  - %d/%d\n', outer_iter, Nouter, inner_iter, ceil(N/blocksize));
      [w deltas alphas] = weighted_subset_svm(feats, labels, weights, 1e-2, ceil(sum(labels==1)*rate));
      clear feats; 

      w_update = w;
      w_update(end+1) = w(end);
      w_update(end-1) = 0; % 

      model = update_model_weights(model, w_update);
   end
end
