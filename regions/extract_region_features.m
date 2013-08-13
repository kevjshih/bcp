function flag = extract_region_features(annotation, dirs)
flag = 1;
if(~exist('dirs', 'var'))
   BDglobals;

   dirs.precomputed_feat = fullfile(dirs.feat_dir, 'tc2');
   
   if(~exist(region_feat_dir, 'file'))
      mkdir(region_feat_dir);
   end
end
dirs.precomputed_feat = fullfile(dirs.feat_dir, 'tc2');
if(~exist(dirs.precomputed_feat))
    mkdir(dirs.precomputed_feat);
end
region_feat_dir = fullfile(dirs.feat_dir, 'region');

[dk bn dk] = fileparts(annotation.filename); 
region_file = fullfile(dirs.region_dir, [bn '_proposals.mat']);

%if(exist(fullfile(region_feat_dir, [bn '_regfeat2.mat'])))
if(exist(fullfile(region_feat_dir, [bn '_regfeat.mat'])))
   return;
end

if(~exist(region_file, 'file'))
   flag = 0;
   return;
end
load(region_file, 'ranked_regions', 'superpixels');
ranked_regions = ranked_regions(1:min(end,500));

tc_file = fullfile(dirs.precomputed_feat, [bn '_tc.mat']);

if(~exist(tc_file, 'file')) % Extract the
    im = imread(fullfile(dirs.im_dir, annotation.filename));
   col = load(fullfile(dirs.feat_dir, '../../', 'colorClusters.mat'));
   tex = load(fullfile(dirs.feat_dir, '../../', 'textonClusters.mat'));
   [textonim, colorim] = processIm2ColorTexture(im, col, tex);
   save(tc_file, 'textonim', 'colorim');
   image_data.textonim = textonim;
   image_data.colorim = colorim;
else
   image_data = load(tc_file);
end

region_boxes = regions2boxes(ranked_regions(1:min(end,500)), superpixels);

region_boxes(:, [1 3]) = min(max(round(region_boxes(:, [1 3])), 1), size(superpixels,2));
region_boxes(:, [2 4]) = min(max(round(region_boxes(:, [2 4])), 1), size(superpixels,1));

nsp = max(superpixels(:));

% Preprocess data
c_hist_sp = double(getRegionHistogram(superpixels, image_data.colorim, 128))';
t_hist_sp = double(getRegionHistogram(superpixels, image_data.textonim, 256))';

for i = 1:length(ranked_regions)
   inds = false(nsp, 1);
   inds(ranked_regions{i}) = 1;
   
   eq0 = inds;
   
   % Histogram Features

   c_hist_region = sum(c_hist_sp(:, eq0), 2);
   c_hist{i} = c_hist_region/(sum(c_hist_region)+eps);

   t_hist_region = sum(t_hist_sp(:, eq0), 2);
   t_hist{i} = t_hist_region/(sum(t_hist_region)+eps);

   % Region features
   sub_im = superpixels(region_boxes(i,2):region_boxes(i,4), region_boxes(i,1):region_boxes(i,3));
   mask = ismember(sub_im,ranked_regions{i});
   sub_mask = imresize(double(mask), [64 64], 'nearest');
   
   region_hog{i} = features(padarray(repmat(sub_mask,[1 1 3]), [8 8]), 8);

   % Extract second set of features that preserve aspect ratio
   %newsize = round(80*size(mask)/max(size(mask)));
   %sub_mask2 = imresize(double(mask), newsize, 'nearest');

   % Pad it so it's 80x80 pixels
   %padding = (80 - size(sub_mask2))/2;
   %sub_mask2 = padarray(padarray(sub_mask2, floor(padding), 0, 'pre'), ceil(padding), 0, 'post');
   %region_hog2{i} = features(padarray(repmat(sub_mask2,[1 1 3]), [8 8]), 8); % Extra padding for weird boundary cases
end

%save(fullfile(region_feat_dir, [bn '_regfeat2.mat']), 'c_hist', 't_hist', 'region_hog', 'region_hog2');
save(fullfile(region_feat_dir, [bn '_regfeat.mat']), 'c_hist', 't_hist', 'region_hog');
