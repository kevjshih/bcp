function model = train_poselet_loc(model, D, cached_scores, part_ind)

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

% Begin by finding the highest scoring examples
for i = 1:length(Dpos)
    if(isempty(cached_pos{i}.regions))
        continue;
    end
   boxes = LMobjectboundingbox(Dpos(i).annotation);
   
   [best_ov best_reg] = max(bbox_overlap_mex(boxes, cached_pos{i}.regions), [], 2);
   
   scores{i} = cached_pos{i}.part_scores(best_reg, part_ind);
   scores{i}(best_ov<0.65) = -inf; % Missed this object
   reg_ind{i} = best_reg;
end

scores_sorted = sort(cat(1, scores{:}), 'descend');
score_thresh = scores_sorted(15); % Use boxes with score greater than this threshold

for i = 1:length(Dpos)
   if(~any(scores{i}>score_thresh))
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
      covered = find(ov>=0.75 & scores{i}>score_thresh); 

      DRAW  = 1;
      if(DRAW)
         clf;
         imagesc(im);
         axis off; axis image;
         hold on;
      end

      kp_rect = kp(gt_to_kp(covered)); % To make parfor happy

      for j_ind = 1:length(covered(:)') % For each good ground truth object...
         j = covered(j_ind);
         kp_j = gt_to_kp(j);

         det_box = double(cached_pos{i}.part_boxes(reg_ind{i}(j), (part_ind-1)*4 + (1:4)));
         flipped = cached_pos{i}.part_trans(reg_ind{i}(j), part_ind)==2;
         
         if(DRAW)
            draw_bbox(det_box, c(mod(j-1,7)+1), 'linewidth', 5);
            plot(kp(kp_j).x, kp(kp_j).y, ['x' c(mod(j-1,7)+1)], 'linewidth', 5);
         end

         kp_rect(j_ind) = normalize_keypoints(kp(kp_j), det_box, flipped);
         kp_rect(j_ind).score = scores{i}(j);
      end

      if(DRAW)
         %pause;
      end

      if(~isempty(kp_rect))
        all_kp0{i} = kp_rect;
      end
   end
end

all_kp = cat(2, all_kp0{:});

all_scores = [all_kp.score];


% Organize keypoints according to label

possible_labels = unique(cat(1, all_kp.label));
label_hash = make_hash(possible_labels);

kp_x = inf(numel(possible_labels), numel(all_kp));
kp_y = inf(numel(possible_labels), numel(all_kp));

for i = 1:length(all_kp)
   kp_ind = lookup_hash(all_kp(i).label, label_hash);

   kp_x(kp_ind, i) = all_kp(i).x(:);
   kp_y(kp_ind, i) = all_kp(i).y(:);
end


% Now compute expected point for each
model.part(part_ind).kp.bbox = [0 0 0 0];

for i = 1:length(possible_labels)
    expected_pos(i, :) = [median(kp_x(i, ~isinf(kp_x(i,:)))), median(kp_y(i, ~isinf(kp_y(i,:))))];
%    expected_pos(i, :) = [mean(kp_x(i, ~isinf(kp_x(i,:)))), mean(kp_y(i, ~isinf(kp_y(i,:))))];
   model.part(part_ind).kp.x(i) = expected_pos(i, 1);
   model.part(part_ind).kp.y(i) = expected_pos(i, 2);
   model.part(part_ind).kp.label{i} = possible_labels{i};
end

model.part(part_ind).kp.score = 0;




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
      kp_rect.label{i} = fliplabel(kp_rect.label{i}); %1:2) = 'R ';
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
