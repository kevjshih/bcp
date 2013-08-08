function cached_new = get_best_pos_reg(D, cached_scores, cls)

INCLUDE_NEG = 1;

cached_new = cached_scores;

for i = 1:length(cached_scores)
   all_inds = [];
   if(isempty(cached_scores{i}.labels))
       continue;
   end

   annotation = D(i).annotation; 
   
   boxes = LMobjectboundingbox(annotation, cls);
   %if(any(cached_scores{i}.labels>0)) % positive image
   if(~isempty(boxes))
      %all_inds = [find(cached_scores{i}.labels<=0)];
      all_inds = max(bbox_overlap_mex(cached_scores{i}.regions, boxes), [], 2)<0.5;
      cached_new{i} = prune_cached_scores(cached_scores{i}, all_inds); 

      % Add in the GT boxes
      N = size(boxes,1);
      cached_new{i}.part_boxes(end+[1:N],:) = 0;
      cached_new{i}.part_scores(end+[1:N], :) = 0;
      cached_new{i}.regions(end+[1:N], :) = boxes;
      cached_new{i}.labels(end+[1:N],1) = 1:N;
      cached_new{i}.region_score(end+[1:N],:) = 0;
      cached_new{i}.scores(end+[1:N],1) = 0;
   else
       % Negative image, nothing to do for now
   end
end
