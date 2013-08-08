function [feat_out imind dists] = get_poselet_examples(model, Dpos, cached_pos)

cls = model.cls;

model.thresh = -inf;

im_dir = [];
BDglobals;

poselet_dir = fullfile(ROOTDIR, 'data', 'annotations', cls);

%[Dpos pos_ind] = LMquery(D, 'object.name', cls, 'exact');
%cached_pos = cached_scores(pos_ind);

clear D; % Just so I don't use it by accident

part_ind = find(~[model.part.computed]);
[model.part.computed] = deal(1);
model.part(part_ind).computed = 0;

c = 'rgbcmyk';

% Get exemplar data ....
[im part_bbox] = extract_exemplar_params(model.part(part_ind).name);
D0 = pasc2D_id(im); % Get D structure

bbox = LMobjectboundingbox(D0.annotation);
[dk ok_bbox] = max(bbox_contained(part_bbox, bbox));

kp = load_poselets(poselet_dir, D0.annotation.filename);

% Match everything up
poselet_boxes = cat(1, kp.bbox);
[ov gt_to_kp] = max(bbox_overlap_mex(bbox(ok_bbox, :), poselet_boxes), [], 2);
%covered = find(ov>=0.75);% & scores{i}>score_thresh); 

kp = kp(gt_to_kp);

% Only use the keypoints within the part - this may not work for all candidates (e.g. no relevant poselets)
kp_covered = kp.x>= part_bbox(1) & kp.x <= part_bbox(3) & kp.y >=part_bbox(2) & kp.y <= part_bbox(4);
kp.x(~kp_covered) = [];
kp.y(~kp_covered) = [];
kp.label(~kp_covered) = [];

model.part(part_ind).kp = kp;

% Normalize!
diag = sqrt(sum((part_bbox([3 4]) - part_bbox([1 2])).^2));
cent = 1/2*(part_bbox([3 4]) + part_bbox([1 2]));

model.part(part_ind).kp.x = (model.part(part_ind).kp.x - cent(1))/diag;
model.part(part_ind).kp.y = (model.part(part_ind).kp.y - cent(2))/diag;


parfor i = 1:length(Dpos)
%   fprintf('%d\n', i);
   [feat_out{i} min_dist{i} best_box{i} flipped{i}] = get_best_match_image(Dpos(i).annotation, model.part(part_ind), im_dir, poselet_dir);
end

for i = 1:length(Dpos)
   imind{i} = repmat(i, size(feat_out{i},2),1);
end

%feat_out = [feat_out{:}];
%min_dist = [min_dist{:}];
%flipped = [flipped{:}];

%dists = cat(2, min_dist{:});
dists = min_dist;

return; 

best_box = [best_box{:}];
imind = cat(1, imind{:});


[dk inds] = sort(dists', 'ascend');

for i = 1:length(inds)
   clf;
   %imagesc(im{inds(i)});
   imshow(fullfile(im_dir, Dpos(imind(inds(i))).annotation.filename));
   axis equal, axis off;
   hold on;
   draw_bbox(best_box(:, inds(i))');

   pause;
end



function [feat_out min_dist, best_box flipped] = get_best_match_image(annotation, part_model, im_dir, poselet_dir)
      feat_out = [];
      min_dist = [];
      best_box = [];
      flipped = [];

   boxes = LMobjectboundingbox(annotation);
   %im = imread(fullfile(im_dir, Dpos(i).annotation.filename));
   
   % Load poselet annotations
   kp = load_poselets(poselet_dir, annotation.filename);
   
   if(isempty(kp))
       fprintf('Annotations missing! %s\n', annotation.filename);
   else
      poselet_boxes = cat(1, kp.bbox);
   
      % Match everything up
      [ov gt_to_kp] = max(bbox_overlap_mex(boxes, poselet_boxes), [], 2);
      covered = find(ov>=0.75);% & scores{i}>score_thresh); 

      I = imread(fullfile(im_dir, annotation.filename));


      for j_ind = 1:length(covered) % For each good ground truth object...
         j = covered(j_ind);
         kp_j = gt_to_kp(j);

         [feat0 dist0 box0 flp0] = get_best_poselet_match_app(I, kp(kp_j), part_model);
         if(~isempty(feat0))
            [feat_out(:, end+1) min_dist(:, end+1) best_box(:, end+1) flipped(:, end+1)] = deal(feat0, dist0, box0, flp0);
         end
      end
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
      kp_rect.label{i} = fliplabel(kp_rect.label{i});
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
