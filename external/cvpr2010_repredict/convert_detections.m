function detections = convert_detections(cached_scores)

detections = cell(length(cached_scores), 0); 

for i = 1:length(cached_scores)
   cs = cached_scores{i};
   for j = 1:size(cs.part_scores, 2)
      det_t = [double(cs.part_boxes(:, 4*(j-1) + [1:4])), cs.part_scores(:, j)];
      ok = nms_iou(det_t, 0.8);

      detections{i, j} = [det_t(ok, 1:4) double(cs.part_trans(ok, j)) det_t(ok, end)];
   end
end
