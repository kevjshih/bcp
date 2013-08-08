function [map aps kp_labels rocs] = train_poselet_loc(model, D, cached_scores, part_ind)

cls = model.cls;

model.thresh = -inf;

im_dir = [];
BDglobals;

poselet_dir = fullfile(ROOTDIR, 'data', 'annotations', cls);


[Dpos pos_ind] = LMquery(D, 'object.name', cls, 'exact');
cached_pos = cached_scores(pos_ind);

clear D; % Just so I don't use it by accident

[model.part.computed] = deal(1);
model.part(part_ind).computed = 0;

c = 'rgbcmyk';

% Begin by finding the highest scoring part detection for each example
for i = 1:length(Dpos)
    if(isempty(cached_pos{i}.regions))
        continue;
    end
   boxes = LMobjectboundingbox(Dpos(i).annotation);
   
   [best_ov best_reg] = max(bbox_overlap_mex(boxes, cached_pos{i}.regions), [], 2);
   
   scores{i} = cached_pos{i}.part_scores(best_reg, part_ind);
   %scores{i}(best_ov<0.65) = -inf; % -inf means we missed the part

   obj_found{i} = ones(size(scores{i}));
   obj_found{i}(best_ov<0.65) = 0; % Missed this object

   reg_ind{i} = best_reg;
end

%scores_sorted = sort(cat(1, scores{:}), 'descend');
score_thresh = -100000000; % A very small but finite number%scores_sorted(30); % Use boxes with score greater than this threshold

for i = 1:length(Dpos)
   if(~any(obj_found{i}==1))
      continue; % Nothing to see here
   end

   fprintf('%d\n', i);
   boxes = LMobjectboundingbox(Dpos(i).annotation);
   im = imread(fullfile(im_dir, Dpos(i).annotation.filename));
   
   % Load poselet annotations
   kp = load_poselets(poselet_dir, Dpos(i).annotation.filename);
   
   if(isempty(kp))
       fprintf('Annotations missing! %s\n', Dpos(i).annotation.filename);
   else
      poselet_boxes = cat(1, kp.bbox);

   
      % Collect part detections
      %hyp{i} = inference(im, model, boxes);
   
      % Match everything up
      [ov gt_to_kp] = max(bbox_overlap_mex(boxes, poselet_boxes), [], 2);
      covered = find(ov>=0.75 & obj_found{i}==1); 

      DRAW  = 1;
      if(DRAW)
         clf;
         imagesc(im);
         axis off; axis image;
         hold on;
      end

      %kp_rect = kp(gt_to_kp(covered)); % To make parfor happy
      kp_pred = kp(gt_to_kp(covered));
      obj_box = [];
      kp_gt = kp(gt_to_kp(covered));

      for j_ind = 1:length(covered(:)') % For each good ground truth object...
         j = covered(j_ind);
         kp_j = gt_to_kp(j);

         det_box = double(cached_pos{i}.part_boxes(reg_ind{i}(j), (part_ind-1)*4 + (1:4)));
         flipped = cached_pos{i}.part_trans(reg_ind{i}(j), part_ind)==2;
         
         if(DRAW)
            draw_bbox(det_box, c(mod(j-1,7)+1), 'linewidth', 5);
            plot(kp(kp_j).x, kp(kp_j).y, ['x' c(mod(j-1,7)+1)], 'linewidth', 5);
         end
         
         % Predict          
         kp_pred(j_ind) = predict_keypoints(model.part(part_ind).kp, det_box, flipped);
         kp_pred(j_ind).score = scores{i}(j);
         obj_box(j_ind, :) = boxes(j, :);

         kp_gt(j_ind) = kp(kp_j);
         %kp_rect(j_ind) = normalize_keypoints(kp(kp_j), det_box, flipped);
         %kp_rect(j_ind).score = scores{i}(j);
      end

      if(DRAW)
         %pause;
      end

      if(~isempty(kp_gt))
        all_kp_pred{i} = kp_pred;
        all_kp_gt{i} = kp_gt;
        all_obj_box{i} = obj_box;
      end
   end
end

all_pred = cat(2, all_kp_pred{:});
all_gt = cat(2, all_kp_gt{:});
all_boxes = cat(1, all_obj_box{:});

all_scores = [all_pred.score];
all_scales = sqrt(sum((all_boxes(:, [3 4]) - all_boxes(:, [1 2])).^2, 2));

% Organize keypoints according to label

possible_labels = unique(cat(1, all_gt.label));
label_hash = make_hash(possible_labels);

kp_pred_x = inf(numel(possible_labels), numel(all_pred));
kp_pred_y = inf(numel(possible_labels), numel(all_pred));

for i = 1:length(all_pred)
   kp_ind = lookup_hash(all_pred(i).label, label_hash);

   kp_pred_x(kp_ind, i) = all_pred(i).x(:);
   kp_pred_y(kp_ind, i) = all_pred(i).y(:);
end

kp_gt_x = inf(numel(possible_labels), numel(all_gt));
kp_gt_y = inf(numel(possible_labels), numel(all_gt));

for i = 1:length(all_gt)
   kp_ind = lookup_hash(all_gt(i).label, label_hash);

   kp_gt_x(kp_ind, i) = all_gt(i).x(:);
   kp_gt_y(kp_ind, i) = all_gt(i).y(:);
end


pred_error = sqrt((kp_gt_x - kp_pred_x).^2 + (kp_gt_y - kp_pred_y).^2);
pred_error_norm = bsxfun(@rdivide, pred_error, all_scales');

relevant = ~isinf(kp_gt_x) & ~isinf(kp_gt_y);

[all_scores_sort b] = sort(all_scores, 'descend');
a = all_scores_sort; % just in case I used this somewhere

th = 0.10; % Needs to be predicted within 10% of box diagonal

p_all = cumsum(pred_error_norm(:, b)<=th, 2)./(cumsum(relevant(:, b), 2)+eps);
r_all = bsxfun(@rdivide, cumsum(pred_error_norm(:, b)<=th, 2), sum(relevant(:, b), 2)+eps);

aps = ap_md(r_all, p_all);
map = mean(aps);
kp_labels = possible_labels;
rocs.p_all = p_all;
rocs.r_all = r_all;
rocs.scores = all_scores_sort; 
rocs.labels = 2*double(pred_error_norm(:, b)<=th)-1;
rocs.labels(relevant(:, b)) = 0; % Irrelevant
return;



if(1 || DRAW)
clf;
hold on;
% Now compute expected point for each

% Now compute expected point for each
for i = 1:length(possible_labels)
    expected_pos(i, :) = [median(kp_x(i, ~isinf(kp_x(i,:)))), median(kp_y(i, ~isinf(kp_y(i,:))))];
%    expected_pos(i, :) = [mean(kp_x(i, ~isinf(kp_x(i,:)))), mean(kp_y(i, ~isinf(kp_y(i,:))))];
    variance(i, :) = [var(kp_x(i, ~isinf(kp_x(i,:)))), var(kp_y(i, ~isinf(kp_y(i,:))))];

    covariance = [cov(kp_x(i, ~isinf(kp_x(i,:))),kp_y(i, ~isinf(kp_y(i,:))))];
    %plot(kp_x(i,:), -kp_y(i,:), [c(mod(i-1,7)+1) 'o']);
%    ellipse(sqrt(variance(i,1)), sqrt(variance(i,2)), 0, expected_pos(i, 1), -expected_pos(i, 2), c(mod(i-1,7)+1));
    error_ellipse(covariance, [expected_pos(i, 1), -expected_pos(i, 2)], 'style', c(mod(i-1,7)+1));
    plot(expected_pos(i,1), -expected_pos(i,2), [c(mod(i-1, 7)+1) 'x'], 'linewidth', 3);
    text(expected_pos(i,1)+.01, -expected_pos(i,2), possible_labels{i});
end

keyboard
return;
for i = 1:length(possible_labels)
    expected_pos(i, :) = [median(kp_x(i, ~isinf(kp_x(i,:)))), median(kp_y(i, ~isinf(kp_y(i,:))))];
    
    ex{i} = plot(kp_x(i,:), -kp_y(i,:), [c(mod(i-1,7)+1) 'o']);
    plot(expected_pos(i,1), -expected_pos(i,2), [c(mod(i-1, 7)+1) 'x'], 'linewidth', 3);
    text(expected_pos(i,1)+.01, -expected_pos(i,2), possible_labels{i});
end
end


%keyboard
% Now test it out


% Compute cumulative average distances for each poselet
[a b] = sort(all_scores, 'descend');
all_scores_s = all_scores(b);
kp_xs = kp_x(:, b);
kp_ys = kp_y(:, b);

for i = 1:length(possible_labels)
   missing = isinf(kp_xs(i, :));
   
   dist_mat0 = squareform(pdist([kp_xs(i,:); kp_ys(i,:)]'));
   dist_mat0(missing, :) = 0; 
   dist_mat0(:, missing) = 0; 
   
   cum_dists = cumsum(cumsum(dist_mat0,1), 2);
   %cum_missing = cumsum(missing).*[1:length(missing)];
   d = ones(length(missing));
   d(missing, :) = 0;
   d(:, missing) = 0;
   
   cum_used = cumsum(cumsum(d,1), 2);
   
   avg_dist(:,i) = diag(cum_dists)./diag(cum_used);
end


function kp_out = predict_keypoints(kp, bbox, flip)

kp_out = kp;

cent = 1/2*(bbox([3 4]) + bbox([1 2]));
scale = sqrt(sum((bbox([3 4]) - bbox([1 2])).^2));


if(flip)
   kp_out.x = -kp.x*scale + cent(1);
   
   for i = 1:length(kp_out.label)
       kp_out.label{i} = fliplabel(kp_out.label{i});
   end
else
   kp_out.x = kp.x*scale + cent(1);
end

kp_out.y = (kp.y*scale + cent(2));

function bbox = get_poselet_bbox(dat)

   bdat = dat.annotation.visible_bounds.Attributes;

   h = str2num(bdat.height);  
   w = str2num(bdat.width);  
   x = str2num(bdat.xmin);  
   y = str2num(bdat.ymin);  

   bbox = [x y x+w y+h];

function kp_rect = normalize_keypoints(kp, bbox, flip)

kp_rect = kp;

cent = 1/2*(bbox([3 4]) + bbox([1 2]));
scale = sqrt(sum((bbox([3 4]) - bbox([1 2])).^2));

if(flip)
   kp_rect.x = -(kp.x - cent(1))/scale;
   
   for i = 1:length(kp_rect.label)
      if(strmatch('L ', kp_rect.label{i}))
         kp_rect.label{i}(1:2) = 'R ';
      elseif(strmatch('R ', kp_rect.label{i}))
         kp_rect.label{i}(1:2) = 'L ';
      end
   end
else
   kp_rect.x = (kp.x - cent(1))/scale;
end
kp_rect.y = (kp.y - cent(2))/scale;
kp_rect.score = 0;


function kp = load_poselets(poselet_dir, im_name)


   [dc bn] = fileparts(im_name);

   ann_files = dir(fullfile(poselet_dir, [bn '_*.xml']));

   if(isempty(ann_files))
       kp = [];
       return;
   end
   
   for j = 1:length(ann_files)
      dat = xml2struct(fullfile(poselet_dir, ann_files(j).name));
      kp(j).bbox = get_poselet_bbox(dat);
      [kp(j).x kp(j).y dc kp(j).label] = get_kp(dat);
      kp(j).score = 0;
   end

function [x y label label_long] = get_kp(dat0)

if(~isfield(dat0.annotation.keypoints, 'keypoint'))
    dat = {};
else
    dat = dat0.annotation.keypoints.keypoint;
end

if(~iscell(dat))
   dat = {dat};
end

x = zeros(length(dat), 1);
y = zeros(length(dat), 1);
label = cell(length(dat), 1);
label_long = cell(length(dat), 1);

for i = 1:length(dat)
   str = dat{i}.Attributes;
   x(i) = str2num(str.x);  
   y(i) = str2num(str.y);  
   lab0 = strrep(str.name, '_', ' ');
   label{i} = lab0(regexp(lab0, '[A-Z]')); % Only use the capital letters (hopefully this is consistent
   label_long{i} = lab0;
end
