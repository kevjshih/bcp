function scores = classify_loo(feats, w_noloo, w_loo, imind)

scores = zeros(size(feats, 2), 1);

for i = 1:length(w_loo)
   todo = imind==i;
   if(isempty(w_loo{i})) % No loo estimate
      cur_w = w_noloo;
   else
      cur_w = w_loo{i};
   end
   
   scores(todo) = cur_w(1:end-1)*feats(:, todo) + cur_w(end);
end
