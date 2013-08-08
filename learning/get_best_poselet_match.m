function [feat_out min_dist_out best_box flipped] = initialize_goalsize_model(I, kp, model)
init_params.sbin = 8;
init_params.hg_size = [8 8];
init_params.MAXDIM = 10;

weight = 0.5; % Penalty paid for a missing keypoint


ARTPAD = 0;
I_real_pad = I;%pad_image(I, ARTPAD);

%Get the hog feature pyramid for the entire image
%p.lpo = 20;
%[f_real, scales] = featpyramid2(I, init_params.sbin, p);
[f_real, scales] = IEfeatpyramid(I, init_params.sbin, 10);


% Setup keypoints
[m_sorted m_ind] = sort(model.kp.label);
m_kp_todo = m_ind(ismember(m_sorted, kp.label));

[gt_sort gt_ind] = sort(kp.label);
gt_kp_todo = gt_ind(ismember(gt_sort, model.kp.label));

% Setup flipped keypoints
kp_t = model.kp;
for i = 1:length(model.kp.label)
   if(strmatch('L ', model.kp.label{i}))
      kp_t.label{i}(1:2) = 'R ';
   elseif(strmatch('R ', model.kp.label{i}))
      kp_t.label{i}(1:2) = 'L ';
   end
end

[m_sorted_flip m_ind_flip] = sort(kp_t.label);
m_kp_todo_flip = m_ind_flip(ismember(m_sorted_flip, kp.label));

[gt_sort_flip gt_ind_flip] = sort(kp.label);
gt_kp_todo_flip = gt_ind_flip(ismember(gt_sort_flip, kp_t.label));

kp_gt(:, 1) = kp.x; % (gt_kp_todo); we'll do the indexing later
kp_gt(:, 2) = kp.y; %(gt_kp_todo);

if(isempty(gt_kp_todo) && isempty(gt_kp_todo_flip))
    min_dist_out = []; %[inf inf];
    feat_out = [];
    best_box = [];
    flipped = [];
    return;
end
min_dist = inf;

procrust_scale = eps + 1/length(gt_ind)*sqrt(sum((kp_gt(:, 1) - mean(kp_gt(:,1))).^2) + sum((kp_gt(:, 2) - mean(kp_gt(:,2))).^2));

keypoint_match = length(m_ind) - length(gt_kp_todo); % Pay some constant penalty for missing keypoints
keypoint_match_flip = length(m_ind) - length(gt_kp_todo_flip); % Pay some constant penalty for missing keypoints

padx = ceil(model.size(2)/2);
pady = ceil(model.size(1)/2);


[x y] = meshgrid(1:size(f_real{1},2)+padx, 1:size(f_real{1},1)+pady);


for i = 1:length(f_real)
   f_real{i} = padarray(f_real{i}, [pady padx 0]);
   xs = x(1:size(f_real{i},1)-model.size(1), 1:size(f_real{i},2)-model.size(2));
   ys = y(1:size(f_real{i},1)-model.size(1), 1:size(f_real{i},2)-model.size(2));

   boxes = rootbox_trans(xs(:), ys(:), init_params.sbin/scales(i), [0 0], 0, padx, pady, model.size(1:2));
   % Overlay expected position from part model
   centers = 1/2*(boxes(:, [3 4]) + boxes(:, [1 2]));
   scale = sqrt(sum((boxes(1, [3 4]) - boxes(1, [1 2])).^2, 2)); % Every box from a scale has the same scale...


   pred_kpX = bsxfun(@plus, centers(:, 1), model.kp.x(m_kp_todo)'*scale);
   pred_kpX_flip = bsxfun(@plus, centers(:, 1), -model.kp.x(m_kp_todo_flip)'*scale);

   pred_kpY = bsxfun(@plus, centers(:, 2), model.kp.y(m_kp_todo)'*scale);
   pred_kpY_flip = bsxfun(@plus, centers(:, 2), model.kp.y(m_kp_todo_flip)'*scale); % Flipped because we need different labels

   procust = sqrt(sum(bsxfun(@minus, pred_kpX, kp_gt(gt_kp_todo, 1)').^2 + ...
                      bsxfun(@minus, pred_kpY, kp_gt(gt_kp_todo, 2)').^2, 2))/procrust_scale + weight*keypoint_match;

   procust_flip = sqrt(sum(bsxfun(@minus, pred_kpX_flip, kp_gt(gt_kp_todo_flip, 1)').^2 + ...
                           bsxfun(@minus, pred_kpY_flip, kp_gt(gt_kp_todo_flip, 2)').^2, 2))/procrust_scale + weight*keypoint_match_flip;
   
   [min_dist_t ind] = min(procust);

   if(min_dist_t < min_dist)
      best_ind = [xs(ind), ys(ind), i 0];
      best_box = boxes(ind, :);
      min_dist = min_dist_t;
   end

   [min_dist_t ind] = min(procust_flip);

   if(min_dist_t < min_dist)
      best_ind = [xs(ind), ys(ind), i 1];
      best_box = boxes(ind, :);
      min_dist = min_dist_t;
   end
end



procrust_dist = min_dist;%/procrust_scale;

min_dist_out = [procrust_dist keypoint_match];

% Now return the features
feat_out = f_real{best_ind(3)}(best_ind(2):best_ind(2)+model.size(1)-1, best_ind(1):best_ind(1)+model.size(2)-1, :);

if(best_ind(4)==1)
   feat_out = flipfeat(feat_out);
end
flipped = best_ind(4);



DISP = 0;
if(DISP)
   % First, display the exemplar keypoints
   figure(1)
   clf;
   %subplot(2, 1, 1)
   plot(model.kp.x, -model.kp.y, '.r', 'linewidth', 3);
   hold on;
   text(model.kp.x, -model.kp.y+0.01, model.kp.label);
   axis equal;

   % Now display the predicted results and the ground truth
   %subplot(2, 1, 2)
   figure(2);
   clf;
   imagesc(I)
   hold on;
   axis image, axis off
   % Repredict everything from scratch (as a sanity check)
   pred_box = rootbox_trans(best_ind(1), best_ind(2), init_params.sbin/scales(best_ind(3)), [0 0], 0, padx, pady, model.size(1:2)); 

   draw_bbox(pred_box);

   if(flipped)
      kp_todo = gt_kp_todo_flip;
   else
      kp_todo = gt_kp_todo;
   end
   kp_missed = 1:size(kp_gt,1);
   kp_missed(kp_todo) = [];

   % Show which GT keypoints were found and which were missed
   plot(kp_gt(kp_todo, 1), kp_gt(kp_todo, 2), '.g', 'linewidth', 4);
   plot(kp_gt(kp_missed, 1), kp_gt(kp_missed, 2), 'xr', 'linewidth', 2);

   % Show predicted keypoint locations
   center = 1/2*(pred_box(:, [3 4]) + pred_box(:, [1 2]));
   scale = sqrt(sum((pred_box([3 4]) - pred_box([1 2])).^2, 2)); % Every box from a scale has the same scale...
  
   if(flipped)
      pred_kpX = bsxfun(@plus, center(:, 1), -model.kp.x(m_kp_todo_flip)'*scale);
      pred_kpY = bsxfun(@plus, center(:, 2), model.kp.y(m_kp_todo_flip)'*scale); % Flipped because we need different labels
   else
      pred_kpX = bsxfun(@plus, center(:, 1), model.kp.x(m_kp_todo)'*scale);
      pred_kpY = bsxfun(@plus, center(:, 2), model.kp.y(m_kp_todo)'*scale);
   end 

   % Predicted keypoints
   plot(pred_kpX, pred_kpY, '.b', 'linewidth', 4); 

   % Lines connecting keypoints with their ground truth counterparts (counterkeypoint?)
   plot([pred_kpX(:), kp_gt(kp_todo, 1)]', [pred_kpY(:), kp_gt(kp_todo, 2)]', 'b');

   % Show the hog features as well
   %subplot(1, 3, 3)
   %visualizeHOG(feat_out);
   keyboard
end

feat_out = feat_out(:);


