function [score, rank] = greedy_max_score(features, overlaps, w_ov, w_feat)
% features - proposal x feat, unary features
% overlaps - proposal x proposal 
% w1 - feat x 1
% w2 - 1 x 1 

if(~exist('w_feat', 'var'))
   w_feat = ones(size(features,2),1);
end

unary_scores = (features*w_feat); % proposal x 1

n_prop = size(features, 1);

f2 = zeros(n_prop, 1);

ith_ind = zeros(n_prop, 1);
rank = zeros(n_prop, 1);


%fout = zeros(numel(w_feat)+1,1);

for i = 1:n_prop
   [score(i) ind_i] = max(unary_scores - w_ov*f2);

%   fout = fout + [features(:,ind_i); w_i*f2(ind_i); w_i*f3(ind_i); f2(ind_i)/n_prop; f3(ind_i)/n_prop];

   f2(ind_i) = Inf;

   ith_ind(i) = ind_i;
   rank(ind_i) = i;

   % Update overlap penalty
   f2 = max(f2, overlaps(:, ind_i));
end

score(ith_ind) = score;

