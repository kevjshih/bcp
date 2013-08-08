function extract_box_features(D, cached_scores, use_disk)

BDglobals;

%if(~exist'
region_feat_dir = fullfile(dirs.feat_dir, 'box');
if(~exist(region_feat_dir, 'file'))
   mkdir(region_feat_dir);
end

dirs.region_feat_dir = region_feat_dir;

parfor_progress(length(D));
parfor i = 1:length(D)
   parfor_progress();
   extract_features(D(i).annotation, cached_scores{i}, dirs);
end
parfor_progress(0);

function extract_features(annotation, cached_scores, dirs)

region_feat_dir = dirs.region_feat_dir;

[dk bn dk] = fileparts(annotation.filename); 
fname = fullfile(region_feat_dir, [bn '_boxfeat.mat']);

if(1||~exist(fname, 'file'))
   % For each box, crop image, extract features
   im = im2double(imread(fullfile(dirs.im_dir, annotation.filename)));

   if(size(im,3)==1)
      im = repmat(im, [1 1 3]);
   end
   box_feat = {};
   for i = 1:size(cached_scores.regions,1)
      reg = round(cached_scores.regions(i, :));
   
      reg([1 3]) = min(size(im,2), max(1, reg([1 3])));   
      reg([2 4]) = min(size(im,1), max(1, reg([2 4])));   
   
      im_cr = im(reg(2):reg(4), reg(1):reg(3), :);
      im_resize = imresize(im_cr, [80 80]);
      box_feat{i} = features(im_resize, 8);

      %newsize = round(80*size(im)/max(size(im,1), size(im,2)));
      %sub_im = imresize(im_cr, newsize(1:2));

      % Pad it so it's 80x80 pixels
      %padding = (80 - size(sub_im))/2;
      %padding(3) = 0;
      %sub_im = padarray(padarray(sub_im, floor(padding), 0, 'pre'), ceil(padding), 0, 'post');
      %box_feat2{i} = features(sub_im, 8); % Extra padding for weird boundary cases
   end

   save(fname, 'box_feat');
end

