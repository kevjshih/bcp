function poselet_consistency(model, D, part_ind)

cls = model.cls;

model.thresh = -inf;

im_dir = [];
BDglobals;

poselet_dir = fullfile(ROOTDIR, 'data', 'annotations', cls);


Dpos = LMquery(D, 'object.name', cls, 'exact');
clear D; % Just so I don't use it by accident

[model.part.computed] = deal(1);
model.part(part_ind).computed = 0;


c = 'rgbcmyk';

for i = 1:length(Dpos)
   fprintf('%d\n', i);
   boxes = LMobjectboundingbox(Dpos(i).annotation);
   im = imread(fullfile(im_dir, Dpos(i).annotation.filename));
   
   % Load poselet annotations
   kp = load_poselets(poselet_dir, Dpos(i).annotation.filename);
   
   if(isempty(kp))
       fprintf('Annotations missing! %s\n', Dpos(i).annotation.filename);
   else
      kp_rect = kp; % To make parfor happy
      poselet_boxes = cat(1, kp.bbox);

   
      % Collect part detections
      hyp{i} = inference(im, model, boxes);
   
      % Match everything up
      [ov kp_to_hyp] = max(bbox_overlap_mex(boxes, poselet_boxes), [], 1);
      missed = ov<=0.75; 

      if(any(missed))
         warning('Some pasc annotation didn''t line up with poselets!\n');
      end


      DRAW  = 0;
      if(DRAW)
         clf;
         imagesc(im);
         hold on;
      end

      for j = 1:size(poselet_boxes,1) % For each poselet...
         hyp_j = kp_to_hyp(j);

         if(DRAW)
            draw_bbox(hyp{i}(hyp_j).bbox(part_ind, :), c(j), 'linewidth', 5);
            plot(kp(j).x, kp(j).y, ['x' c(j)], 'linewidth', 5);
         end

         kp_rect(j) = normalize_keypoints(kp(j), hyp{i}(hyp_j).bbox(part_ind, :), hyp{i}(hyp_j).loc(part_ind, 4)==2);
         kp_rect(j).score = hyp{i}(hyp_j).score(part_ind);
      end

      if(DRAW)
         pause;
      end

      all_kp0{i} = kp_rect;
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

keyboard
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
