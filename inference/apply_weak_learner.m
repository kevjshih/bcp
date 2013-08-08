function cached_data = apply_weak_learner(cached_data, new_learner, ...
                                          dopar)
% computes the score for each region using 'new_learner'
% and then stores it in the cached_scores{i}.score field

if(~exist('dopar') || dopar==0)
for i = 1:length(cached_data)
    if(~isempty(cached_data{i}.regions))
   %cached_data{i}.scores = boost_classify(cached_data{i}.part_scores, new_learner);
   %cached_data{i}.scores = boost_classify([cached_data{i}.part_scores [1:size(cached_data{i}.part_scores,1)]'], new_learner);
   if(isfield(cached_data{i}, 'part_scores'))
    cached_data{i}.scores = boost_classify([cached_data{i}.part_scores(1:end, :) cached_data{i}.region_score [1:size(cached_data{i}.part_scores,1)]'], new_learner);
   else
    cached_data{i}.scores = boost_classify([cached_data{i}.region_score [1:size(cached_data{i}.region_score,1)]'], new_learner);
   end
    end
end

else

end
