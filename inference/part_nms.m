function cached_scores = part_nms(cached_scores0, ov_th)

cached_scores = cached_scores0;

parfor i = 1:length(cached_scores)
   fprintf('.');
   if(isempty(cached_scores{i}.regions))
      continue;
   end

   for j = 1:size(cached_scores{i}.regions,1)
      [cached_scores{i}.part_scores(j,:)] = part_nms_helper(cached_scores0{i}.part_scores(j,:), cached_scores0{i}.part_boxes(j,:), ov_th);
   end
end




function [scores] = part_nms_helper(scores0, boxes0, ov)

boxes1 = reshape(boxes0(:), 4, [])';
scores = -inf*ones(1, numel(scores0));
picked = nms_iou(boxes1, ov);

scores(picked) = scores0(picked);
