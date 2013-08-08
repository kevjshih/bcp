function model = train_exemplar_poselet(model, part_ind)
% This very simply extracts the exemplar's poselet locations
cls = model.cls;

model.thresh = -inf;

im_dir = [];
BDglobals;

poselet_dir = fullfile(ROOTDIR, 'data', 'annotations', cls);

if(~exist('part_ind', 'var'))
   part_ind = find(~[model.part.computed]);
   %[model.part.computed] = deal(1);
   %model.part(part_ind).computed = 0;
end
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
kp = kp(gt_to_kp); % Now it only contains the exemplar keypoints


if(0) %Use all keypoints
%kp_rect = normalize_keypoints(kp, bbox, flip);
% Use keypoints from the original exemplar
kp_covered = kp.x>= part_bbox(1) & kp.x <= part_bbox(3) & kp.y >=part_bbox(2) & kp.y <= part_bbox(4);
kp.x(~kp_covered) = [];
kp.y(~kp_covered) = [];
kp.label(~kp_covered) = [];
end

model.part(part_ind).kp = kp;



flipped = 0;

USE_LATENT_POS = 1;

if(USE_LATENT_POS)
    model_t = model; %.part
    [model_t.part.computed] = deal(1);
    model_t.part(part_ind).computed = 0;
    model_t.thresh =  -inf;
   cached_tmp{1}.regions = bbox(ok_bbox, :);
   cached_tmp{1}.labels = 1;
   cached_tmp{1}.scores = 0;
   cached_tmp{1}.part_scores = [];zeros(1, model.num_parts);
   cached_tmp{1}.part_boxes = [];%: [500x0 double]

   [dk exemp_hyp] = collect_training_ex(model_t, D0, cached_tmp, 1);

   if(~isempty(exemp_hyp{1}))
      part_bbox = exemp_hyp{1}.bbox(part_ind, :);
      flipped = exemp_hyp{1}.loc(part_ind, 4)==2;
   end
end

if(flipped)
   warning('This model predicted the highest scoring detection on the exemplar was flipped!  This shouldn''t happen unless it''s really symmetric\n');
end

   % Normalize!
   diag = sqrt(sum((part_bbox([3 4]) - part_bbox([1 2])).^2));
   cent = 1/2*(part_bbox([3 4]) + part_bbox([1 2]));

   if(flipped)
      kp_out = model.part(part_ind).kp;
      kp_out.x = (-kp_out.x - cent(1))/diag;

      for i = 1:length(kp_out.label)
         kp_out.label{i} = fliplabel(kp_out.label{i});
      end

      model.part(part_ind).kp = kp_out;
   else
      model.part(part_ind).kp.x = (model.part(part_ind).kp.x - cent(1))/diag;
   end
   model.part(part_ind).kp.y = (model.part(part_ind).kp.y - cent(2))/diag;


return; 
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
function bbox = get_poselet_bbox(dat)

   bdat = dat.annotation.visible_bounds.Attributes;

   h = str2num(bdat.height);  
   w = str2num(bdat.width);  
   x = str2num(bdat.xmin);  
   y = str2num(bdat.ymin);  

   bbox = [x y x+w y+h];



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
