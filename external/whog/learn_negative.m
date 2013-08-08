function learn_negative

% Use pascal train set
addpath(genpath('~/prog/VOC2010/VOCdevkit/VOCcode'));
addpath('~/prog/voc-release3.1');
VOCopts = [];
VOCinit;

ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');


sbin = 8; interval = 5; % Subsampling pyramid

mean_cell = zeros(1, 1, 31);

parfor i = 1:length(ids)
   fprintf('%d/%d\n', i, length(ids));
   im = imread(sprintf(VOCopts.imgpath, ids{i}));

   pyr = featpyramid(im, sbin, interval);
   
   im_mean_cell = zeros(1, 1, 31);
   cell_count = 0;

   for j = 1:length(pyr)
      im_mean_cell = im_mean_cell + sum(sum(pyr{j},1),2);
      cell_count = cell_count + size(pyr{j},1)*size(pyr{j},2);
   end
  
   
   
   mean_cell = mean_cell + im_mean_cell/cell_count;
  
end


mean_cell = mean_cell/length(ids);

bg_cell = mean_cell;

mkdir('data');
save('data/bg_cell.mat', 'bg_cell');
