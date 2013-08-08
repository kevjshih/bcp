function doit(model, year)

[model.part.computed] = deal(0);

im_dir = [];
BDglobals;

im_dir = strrep(im_dir, '2010', year); % This is hacky...

out_dir = fullfile('data/tmp/detections/', year, model.cls);
mkdir(out_dir)

d = dir([im_dir '/*.jpg']);

parfor i = 1:length(d)
   fprintf('%d/%d\n', i, length(d));
   helper(d(i).name, model, im_dir, out_dir);
end



function helper(filename, model, im_dir, out_dir)

[dk bn] = fileparts(filename);

fname = fullfile(out_dir, [bn '_part_det.mat']);


if(~exist(fname, 'file'))
   im = im2double(imread(fullfile(im_dir, filename)));

   [boxes flipped] = inference_part(im, model, 1); % can apply detection type nms later
   clf;
   imagesc(im);
   hold on;
   draw_bbox(boxes{1}, 'linewidth', 3);
   pause
   save(fname, 'boxes', 'flipped');
else
   fprintf('Already computed %s\n', fname);
end

