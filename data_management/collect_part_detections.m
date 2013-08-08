function [ids, scores, boxes, regions] = collect_part_detections(model, D, cached_scores, part_ind)

if(~exist('part_ind', 'var'))
   part_ind = 1;
end


for i = 1:length(D)
   [dk ids{i}] = fileparts(D(i).annotation.filename);

   scores{i} = cached_scores{i}.scores(:, part_ind);
   boxes{i} = cached_scores{i}.part_boxes(:, (part_ind-1)*4 + [1:4]);
   regions{i} = cached_scores{i}.regions;
end
