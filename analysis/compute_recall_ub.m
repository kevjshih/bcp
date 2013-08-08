function [recall best_overlaps] = compute_recall_ub(D, cached_scores, cls)
% Compute maximum possible recall for this category with given cached regions

[Dpos inds] = LMquery(D, 'object.name', cls, 'exact');
cached_sub = cached_scores(inds);

for i = 1:length(Dpos)    
   boxes = LMobjectboundingbox(Dpos(i).annotation, cls);
   
   if(~isempty(cached_sub{i}.regions))
      [best_overlaps{i} best_ind] = max(bbox_overlap_mex(boxes, cached_sub{i}.regions), [], 2);
   else
      best_overlaps{i} = zeros(size(boxes,1), 1);
   end
end

all_overlaps = cat(1, best_overlaps{:});
recall = mean(all_overlaps>=0.5);
