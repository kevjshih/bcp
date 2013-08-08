function rocs = test_bow(D, cached_scores, cls, bow_scores, N)


for i = 1:N
   for j = 1:length(cached_scores)
      cached_scores{j}.scores = bow_scores{j}(:, i);
   end

   rocs(i) = test_given_cache(D, cached_scores, cls, 0.5);
end
