function cached_new = get_best_pos_reg(D, cached_scores, cls)
% Copies scores from closest to GT, plus GT box

INCLUDE_NEG = 1;

cached_new = cached_scores;

for i = 1:length(cached_scores)
   all_inds = [];
   if(isempty(cached_scores{i}.labels))
       continue;
   end

   annotation = D(i).annotation; 
   if(any(cached_scores{i}.labels>0)) % positive image
      boxes = LMobjectboundingbox(annotation, cls);
      [overlaps best_ind] = max(bbox_overlap_mex(boxes, cached_scores{i}.regions), [], 2);
 
      pos_inds = best_ind(overlaps>=0.5);
      all_inds = [pos_inds; find(cached_scores{i}.labels<=0)];

      cached_new{i} = prune_cached_scores(cached_scores{i}, all_inds); 
      % Transfer gt boxes over
      cached_new{i}.regions(1:length(pos_inds), :) = boxes(overlaps>=0.5, :);
   else
       % Negative image, nothing to do for now
   end
end
