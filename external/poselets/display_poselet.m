function display_poselet(D, cls)

BDglobals;

[dk inds] = LMquery(D, 'object.name', cls, 'exact');

poselet_dir = fullfile(ROOTDIR, 'data', 'annotations', cls);


colors = 'rgbcmyk';

for i = inds(:)'
   clf;
   im = imread(fullfile(im_dir, D(i).annotation.filename));
   imagesc(im);
   hold on;
   [dc bn] = fileparts(D(i).annotation.filename);

   ann_files = dir(fullfile(poselet_dir, [bn '_*.xml']));


   for j = 1:length(ann_files)
      col = colors(mod(j-1, length(colors))+1);

      dat = xml2struct(fullfile(poselet_dir, ann_files(j).name));

      bbox = get_bbox(dat);
      [x y label label_long] = get_kp(dat);

      draw_bbox(bbox, col);
      
      for k = 1:length(x)
         text(x(k), y(k), label{k}, 'backgroundcolor', col);
      end      
   end

   drawnow;
   pause;%(0.1);
end


function bbox = get_bbox(dat)

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
