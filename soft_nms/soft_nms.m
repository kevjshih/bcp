function cached_scores = soft_nms(cached_scores, weight)

for i = 1:length(cached_scores)
   if(mod(i, ceil(length(cached_scores)/100))==0)
      fprintf('.');
   end

   if(isempty(cached_scores{i}.labels))
      cached_scores{i}.nms_scores = [];
   else
     cached_scores{i}.nms_scores = reshape(greedy_nms(cached_scores{i}.scores, cached_scores{i}.overlaps, weight), [], 1);
   end
end

fprintf('\n');
