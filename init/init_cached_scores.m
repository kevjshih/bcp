function [cached_scores] = mine_examples(model, D)

model.thresh = -inf; % Don't want to prune examples

dirs = [];
BDglobals;

for ex = 1:length(D)
   fprintf('%d/%d\n', ex, length(D));
   [cached_scores{ex}] = mine_example(model, D(ex).annotation, dirs); 
end

function [cached_scores] = mine_example(model, annotation, dirs)
   % To define: region_dir, feat_dir, im_dir
   cls = model.cls;

   [dk bn dk] = fileparts(annotation.filename); 
   feat_file = fullfile(dirs.feat_dir, [bn '.mat']);
   label_dir = fullfile(dirs.label_dir, cls);
   label_file = fullfile(label_dir, [bn '.mat']);
   region_file = fullfile(dirs.region_dir, [bn '_proposals.mat']);

   try
      error(''); 
      load(label_file, 'regions', 'labels');
   catch % Doing it this way since some files managed to get corrupted
      if(exist(region_file, 'file'))
         load(region_file, 'ranked_regions', 'superpixels');
   
         regions = regions2boxes(ranked_regions(1:min(end,500)), superpixels);
         labels = regions2labels(regions, annotation, cls);
   
         if(~exist(label_dir, 'file'))
            mkdir(label_dir);
         end
   
         save(label_file, 'regions', 'labels');
      else % Original regions file doesn't exist, no datacached
         cached_scores.regions = [];
         cached_scores.labels = [];
         cached_scores.scores = [];
         cached_scores.part_scores = zeros(0,0);
         cached_scores.part_boxes = zeros(0,0);
         return;
      end
   end
   
   cached_scores.regions = regions;
   cached_scores.labels = labels;
   cached_scores.scores = zeros(size(regions, 1), 1);
   cached_scores.part_scores = zeros(size(regions, 1), 0);
   cached_scores.part_boxes= zeros(size(regions, 1), 0);
