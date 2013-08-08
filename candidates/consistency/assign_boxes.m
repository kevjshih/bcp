function [feats, boxes inds] = assign_boxes(Dpos, boxes0, flipped)

error('This function is incomplete');

for i = 1:length(Dpos)
   bbox_gt = LMobjectboundingbox(Dpos);

   [overlap ind] = max(bbox_contained(boxes0{i}, bbox_gt), [], 2);

   for j = 1:size(bbox_gt, 1)
      ok = find(overlap>=0.8 & ind==j);

      boxes{end+1} = boxes0{i}(ok, :);
      inds{end+1} = [repmat([i j], length(ok), 1), ok(:)];
      
      gt_bbox = bbox_gt(j, :);
      rel_coords = bsxfun(@rdivide, bsxfun(@minus, boxes{end},  gt_bbox), [gt_dim gt_dim]);
   
      rel_coords(flipped{i}(ok), :) = bsxfun(@mtimes, [-1 0 -1 0].*rel_coords(flipped{i}(ok), [3 2 1 4])); % Flip L/R coordinates


   end
end
