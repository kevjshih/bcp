function [model w_all] = train_consistency(model, D, cached_gt, C, MAX_ITER)
% Assumes cached_gt is passed in

if(~exist('C', 'var'))
   C = 15;
end

curpart = find(~[model.part.computed]);

model.part(curpart).spat_const = [0 1 0.8 1 0 1];


if(~exist('MAX_ITER', 'var'))
   MAX_ITER = 7;
end

neg_feats = [];

for i = 1:MAX_ITER % Conservative, run for many iterations
   cached_gt_tmp = get_consistent_examples(model, D, cached_gt);

   if(i==MAX_ITER && nargout>=2) % Do loo
      [model neg_feats w_all] = train_loo_cache(model, D, cached_gt_tmp, 10, 2, 1, C, neg_feats);
   else
      [model neg_feats] = train_loo_cache(model, D, cached_gt_tmp, 10, 2, 1, C, neg_feats);
   end
end
