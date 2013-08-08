function cached_scores = merge_results(cached_scores, cached_scores2)


for i = 1:length(cached_scores)
   % Merge part and region results
   cached_scores{i}.part_scores = [cached_scores{i}.part_scores, cached_scores2{i}.part_scores];
   cached_scores{i}.part_boxes = [cached_scores{i}.part_boxes, cached_scores2{i}.part_boxes];
   cached_scores{i}.region_score = [cached_scores{i}.region_score, cached_scores2{i}.region_score];
   cached_scores{i}.labels = [cached_scores{i}.labels cached_scores2{i}.labels];
end
