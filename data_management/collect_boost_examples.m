function [labels scores feats] = collect_boost_examples(D, model, cached_scores)


dirs = [];
BDglobals;
parfor ex = 1:length(D)
   fprintf('%d/%d\n', ex, length(D));
   [labels{ex} feats{ex} hyp{ex}] = mine_example(model, D(ex).annotation, cached_scores, dirs); 
end
