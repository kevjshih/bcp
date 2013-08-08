function [labels] = mine_examples(model, D, cached_scores)

for ex = 1:length(D)
%    parfor_progress;
   labels{ex} = cached_scores{ex}.labels;
end
