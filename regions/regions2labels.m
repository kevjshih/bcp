function labels = regions2labels(regions, ann, cls)

   %obj = get_object_ind(ann);

   if(isempty(ann.object))
       labels = -ones(size(regions,1), 1);
       return;
   end
   
   boxes = LMobjectboundingbox(ann, cls);
   %boxes = boxes(obj, :);
    
   labels = -ones(size(regions,1),1);
   if(~isempty(boxes))
      [overlaps best_ind] = max(bbox_overlap_mex(boxes, regions), [], 1);
  
      good = overlaps>0.5;
      dont_care = ~good & overlaps>0.2;
      labels(good) = best_ind(good);
      labels(dont_care) = 0;
   end


