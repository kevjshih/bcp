function [idx box_centers] = cluster_boxes(boxes, scores, ov)

%overlaps = bbox_overlap_mex(boxes, boxes);
%overlaps(overlaps<ov) = 0;

%scores = (sum(overlaps));
 
[scores_sorted ordering] = sort(scores, 'descend');
boxes_sorted = boxes(ordering, :);

box_centers = zeros(0, 4);
idx_sorted = [];

for iter = 1:10
   % Adjust cluster centers
   box_centers = reestimate_boxes(boxes_sorted, idx_sorted, box_centers, ov);
   % Reassign boxes
   [idx_sorted box_centers] = cluster_iteration(boxes_sorted, box_centers, ov);
   % Prune boxes 
   %box_centers = prune_centers(box_centers, idx_sorted, ov);
   % Reassign again
   %[idx_sorted box_centers] = cluster_iteration(boxes_sorted, box_centers, ov);
   
   x = hist(idx_sorted, 1:max(idx_sorted));
   bar(sort(x));
   pause
end

idx = zeros(size(idx_sorted));
idx(ordering) = idx_sorted; % Unsort list


function box_centers = prune_centers(box_centers, idx, ov)

% Compute counts
counts = hist(idx, 1:max(idx));

[counts_sorted sort_inds] = sort(counts, 'descend'); % Greedy search based on most common center

overlaps = bbox_overlap_mex(box_centers, box_centers);
overlaps = overlaps.*(1-eye(size(overlaps))); % Get rid of diagonal

delete = zeros(size(box_centers,1), 1);

for sind = sort_inds(:)'
   if(~delete(sind)) % If this box hasn't been deleted already ...
      % Find any center that has sufficient overlap
      delete = delete | overlaps(:, sind)>=ov;
   end
end

fprintf('Pruned %d centers: %d->%d\n', sum(delete), length(delete), sum(~delete));
box_centers(delete, :) = [];


function box_centers = reestimate_boxes(boxes, idx, box_centers0, ov_th)

box_centers = box_centers0;

delete = [];

for i = 1:size(box_centers, 1)
   % Find box that maximizes overlap with all others
   thisbox = find(idx==i);

   overlaps = bbox_overlap_mex(boxes(thisbox, :), boxes(thisbox, :));
   overlaps(overlaps<ov_th) = 0;
   %overlaps = double(overlaps>=ov_th);
   
   mean_ov = mean(overlaps, 1);
   [dk best_box] = max(mean_ov);

   box_centers(i, :) = boxes(thisbox(best_box), :);
   if(length(thisbox)<3)
       delete(i) = 1;
   end
end

%box_centers(delete==1, :) = [];

function [box_inds boxes] = cluster_iteration(boxes0, box_centers, ov_th)

Nbox = size(boxes0, 1);

n_used = size(box_centers, 1);
boxes = zeros(Nbox, 4);
boxes(1:n_used, :) = box_centers;
box_inds = zeros(Nbox, 1);


for i = 1:size(boxes)
   if(isempty(boxes))
      % Start by adding first box
      boxes(1, :) = boxes0(i, :); % 
      n_used = 1;
      box_inds(i) = 1;
   else
      [best_ov best_ind] = max(bbox_overlap_mex(boxes0(i, :), boxes(1:n_used, :)), [], 2);

      if(best_ov>=ov_th) % It matches one of the existing boxes
         box_inds(i) = best_ind;
      else % No match, create new one
         fprintf('New box added!\n');
         n_used = n_used + 1;
         box_inds(i) = n_used;
         boxes(n_used, :) = boxes0(i, :);
      end
      %draw_bbox(boxes(1:n_used, :));
   end
%   pause
end

boxes = boxes(1:n_used, :)

