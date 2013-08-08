function [cached_scores] = mine_examples(model, root_model, D)

model.thresh = -inf; % Don't want to prune examples
dirs = [];
BDglobals;

parfor ex = 1:length(D)
   fprintf('%d/%d\n', ex, length(D));
   [cached_scores{ex}] = mine_example(model, root_model, D(ex).annotation, dirs); 
end

function [cached_scores] = mine_example(model, root_model, annotation, dirs)
   % To define: region_dir, feat_dir, im_dir
   cls = model.cls;

   [dk bn dk] = fileparts(annotation.filename); 
   feat_file = fullfile(dirs.feat_dir, [bn '.mat']);
   label_dir = fullfile([dirs.label_dir '_fgmr'], cls);
   label_file = fullfile(label_dir, [bn '.mat']);
   region_file = fullfile(dirs.region_dir, [bn '_proposals.mat']);

   if(0&&exist(label_file, 'file'))
      load(label_file, 'regions', 'labels', 'region_score');
   elseif(exist(region_file, 'file')) % Just checking this so we use the same images as the proposal data
%      regions = regions2boxes(ranked_regions(1:min(end,500)), superpixels);
      im_file = fullfile(dirs.im_dir, annotation.filename);
      im = imread(im_file);

      if(1) % This one seems to work better
         regions = imgdetect_top_k(im, root_model, 1000);
         I = nms_iou(regions, 0.75);
      else
         regions = imgdetect_top_k(im, root_model, 10000);
         I = nms_iou(regions, 0.5);
      end
      region_score = regions(I, 6);
      regions = regions(I, [1:4]);
      labels = regions2labels(regions, annotation, cls);

      if(~exist(label_dir, 'file'))
         mkdir(label_dir);
      end

      %save(label_file, 'regions', 'labels', 'region_score');
   else % Original regions file doesn't exist, no datacached
      cached_scores.regions = [];
      cached_scores.labels = [];
      cached_scores.scores = [];
      cached_scores.region_score = [];
      return;
   end
   
   cached_scores.regions = regions;
   cached_scores.region_score = region_score;
   cached_scores.labels = labels;
   cached_scores.scores = zeros(size(regions, 1), 1);

