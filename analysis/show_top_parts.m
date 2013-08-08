function [roc res] = test_given_cache(D, cached_scores, cls, part_ind)

% -------- Do Not Change ---------- %
areaThresh = 24*24; % this should stay fixed
ovThresh = 0.50;
ovThreshPart = 0.25;
% --------------------------------- %


for i = 1:length(D)
   D(i).annotation.folder = '';
end

BDglobals; % Defines im_dir

col = 'cmykrgb';


[dk good] = LMquery(D, 'object.name', cls, 'exact');
% Select boxes with simple NMS
for i = 1:length(D)
   if(~isempty(cached_scores{i}.scores))
      all_detections = [double(cached_scores{i}.part_boxes(:,4*(part_ind-1) + [1:4])), cached_scores{i}.part_scores(:,part_ind)];
      det_ind = nms_v4(all_detections, 0.8);
      detections = all_detections(det_ind, :);
      boxes{i} = detections(:,1:4);
      scores{i} = detections(:, end);
      detection_list{i} = detections;
      if(0&&ismember(i, good))
         im = LMimread(D, i, im_dir);

         for k = 1:min(5, numel(scores{i}))
            clf;
            imagesc(im);
            hold on;
            draw_bbox(detections(k, :), 'linewidth', 3);

            for j = 1:10
               draw_bbox(cached_scores{i}.part_boxes(det_ind(k), (j-1)*4 + [1:4]), ['--' col(mod(j, end)+1)]);
            end
            pause
         end
      end
   else
      boxes{i} = zeros(0, 4);
      scores{i} = zeros(0,1);
   end

   [dk ids{i}] = fileparts(D(i).annotation.filename);
end

%show_top_detections(detection_list(good), D(good));
show_top_detections(detection_list, D);
