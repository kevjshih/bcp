cached_scores = add_region_overlaps(cached_scores); % Computes overlaps between all regions


% Do a grid search
threshs = linspace(7, 14, 25);
for i = length(threshs):-1:1
   fprintf('===================== %f ==========================\n', threshs(i));
   cached_scores = soft_nms(cached_scores, threshs(i)); % Computes nms_scores
   roc(i) = test_given_cache(D, cached_scores, cls, 0.5, 0, 0);
   fprintf('\n\n\n');
end
