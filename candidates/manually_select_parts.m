function [ims boxes] = manually_select_parts(D, cls)

fname = sprintf('data/%s_part_ids.mat', cls);

if(exist(fname, 'file'))
   load(fname, 'ims', 'boxes');
   return;
end
   

Dsub = LMquery(D, 'object.name', cls, 'exact');
BDglobals;

boxes = {};
ims = {};

for i = 1:length(Dsub)
%   im = LMimread(Dsub, i, im_dir);
   im = imread(fullfile(im_dir, Dsub(i).annotation.filename));
   clf;
   imagesc(im)
   hold on;
   [x y button] = ginput(1);
   if(button~=1)
      continue;
   end

   box = [];

   box(1) = x;
   box(2) = y;
   plot(box(1), box(2), 'x', 'linewidth', 3);
   [x y button] = ginput(1);

   if(button~=1)
      continue;
   end

   box(3) = x;
   box(4) = y;

   boxes{end+1} = box;
   ims{end+1} = Dsub(i).annotation.filename;

   if(length(ims)==5)
      break;
   end
end

save(fname, 'boxes', 'ims');
