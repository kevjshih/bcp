function [boxes flipped] = doit(D, model)

[model.part.computed] = deal(0);

im_dir = [];
BDglobals;


out_dir = ['data/tmp/detections/' model.cls];

if(nargout>0)
   dosave = 0;
else
   mkdir(out_dir)
   dosave = 1;
end

%par
parfor i = 1:length(D)
   fprintf('%d/%d\n', i, length(D));
   [boxes(i, :) flipped(i, :)] = helper(D(i).annotation, model, im_dir, out_dir, dosave);
end



function [boxes flipped] = helper(ann, model, im_dir, out_dir, dosave)

[dk bn] = fileparts(ann.filename);

if(dosave)
   fname = fullfile(out_dir, [bn '_part_det.mat']);

   error('asdf');
   if(~exist(fname, 'file'))
      im = im2double(imread(fullfile(im_dir, ann.filename)));

      [boxes flipped] = inference_part(im, model, 1);

      save(fname, 'boxes', 'flipped');
   end
else
   im = im2double(imread(fullfile(im_dir, ann.filename)));
   [boxes flipped] = inference_part(im, model, 1);
end
