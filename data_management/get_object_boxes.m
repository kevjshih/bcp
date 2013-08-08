function [ims, obj_box] = get_object_boxes(VOCopts, set, cls)
% Returns the bounding boxes for each object instance of the given
% class. Each index into 'ims' and 'obj_box' corresponds to one object.
basedir = fullfile(VOCopts.localdir, 'object_boxes');
if ~exist(basedir, 'dir')
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [set '_' cls '.mat']);
if fileexists(cached_filename)
   fprintf(['Loading object boxes from "' cached_filename '"...\n']);
   load(cached_filename, 'ims', 'obj_box');
   return;
end

% BDglobals must go after load_init_data, otherwise 'im_dir' is incorrect.
load_init_data;
BDglobals;

ims = {};
obj_box = {};

switch set
  case 'train'
    [Dpos inds] = LMquery(D, 'object.name', cls, 'exact');
  case 'test'
    [Dpos inds] = LMquery(Dtest, 'object.name', cls, 'exact');
  otherwise
    disp(['Invalid set: ' set ' (must be train/test)']);
    return;
end

for Dpos_i = 1:size(Dpos, 2)
   im = fullfile(im_dir, Dpos(Dpos_i).annotation.filename);
   bbox = LMobjectboundingbox(Dpos(Dpos_i).annotation, cls);
   for bbox_i = 1:size(bbox, 1)
      ims{end+1} = im;
      obj_box{end+1} = bbox(bbox_i, :);
   end
end

save(cached_filename, 'ims', 'obj_box');
end