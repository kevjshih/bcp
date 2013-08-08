function compare_top_detections(D, cached_scores1, cached_scores2, cls)

BDglobals;

boxes1 = collect_detections(cached_scores1);
boxes2 = collect_detections(cached_scores2);



for i = 1:length(D)
   D(i).annotation.folder = '';
end

%%%%%%%%%%%%%%%% collection 1
all_boxes1 = cat(1,boxes1{:});
all_scores1 = all_boxes1(:,end);

for i = 1:length(boxes1)
   im_ind1{i} = i*ones(size(boxes1{i},1), 1);
end

all_inds1 = cat(1, im_ind1{:});

[dk best_inds1] = sort(all_scores1, 'descend');

%%%%%%%%%%%%%%%%% collection 2
all_boxes2 = cat(1,boxes2{:});
all_scores2 = all_boxes2(:,end);

for i = 1:length(boxes2)
   im_ind2{i} = i*ones(size(boxes2{i},1), 1);
end

all_inds2 = cat(1, im_ind2{:});

[dk best_inds2] = sort(all_scores2, 'descend');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


dims = [5 5];

for i_outer = 1:length(best_inds1)
   imind = all_inds1(best_inds1(i_outer));
   im = LMimread(D, imind, im_dir);

   gtbox = LMobjectboundingbox(D(imind).annotation, cls);

   figure(1)
   clf
   imagesc(im);
   hold on;
   axis image;
   axis off;
   for j = 1:min(5, size(boxes1{imind},1))
      pred_box = boxes1{imind}(j, :);
      if(~isempty(gtbox) && max(bbox_overlap_mex(gtbox, pred_box(1:4)))>0.5)
         color = 'g';
      else
         color = 'r';
      end

      draw_bbox(pred_box, color, 'linewidth', 3);
   end

   
   figure(2)
   clf
   imagesc(im);
   hold on;
   axis image;
   axis off;
   for j = 1:min(5, size(boxes2{imind},1))
      pred_box = boxes2{imind}(j, :);
      if(~isempty(gtbox) && max(bbox_overlap_mex(gtbox, pred_box(1:4)))>0.5)
         color = 'g';
      else
         color = 'r';
      end

      draw_bbox(pred_box, color, 'linewidth', 3);
   end

   pause
end


function detections = collect_detections(cached_scores)

th = 0.5;
% Select boxes with simple NMS
for i = 1:length(cached_scores)
   if(~isempty(cached_scores{i}.scores))
      all_detections = [cached_scores{i}.regions, cached_scores{i}.scores];
      det_ind = nms_v4(all_detections, th);
      detections{i} = all_detections(det_ind, :);
   else
      detections{i} = zeros(0, 5);
   end
end

