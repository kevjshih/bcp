function applicability = manual_set_applicability(VOCopts, part, set, use_cached)
% Returns an applicability of a part to the objects of its class in
% the specified set. Set 'use_cached' to true if you don't want to
% verify the cached applicability.
%
% Use 'applicability' for lookups with 'manual_get_applicability'.

if ~exist('use_cached', 'var')
   use_cached = false;
end

basedir = fullfile(VOCopts.localdir, 'part_applicability');
if ~exist(basedir, 'dir')
   mkdir(basedir);
end

[~, name] = fileparts(part.im);
cached_filename = fullfile(basedir, [set '_' name '_' mat2str(part.bbox) '.mat']);

if fileexists(cached_filename)
   fprintf(['Loading part applicability from "' cached_filename '"...\n']);
   load(cached_filename, 'applicability');
   if use_cached
      return;
   end
else
   [ims obj_box] = get_object_boxes(VOCopts, set, part.cls);
   applicability.ims = ims;
   applicability.obj_box = obj_box;
   applicability.correct = [];
end

for i = 1:length(applicability.ims)
   [~, im_id] = fileparts(applicability.ims{i});
   applicability.ims{i} = sprintf(VOCopts.imgpath, im_id);
end

applicability.correct = ui_check_part(part, applicability.ims, applicability.obj_box, applicability.correct);
applicability.lookup = create_applicability_lookup(applicability.ims, applicability.obj_box, applicability.correct);
save(cached_filename, 'applicability');
end

function lookup = create_applicability_lookup(ims, obj_box, correct)
lookup = [];
for ims_i = 1:length(ims)
   [dc name] = fileparts(ims{ims_i});
   % Only use y-coordinates because x may be flipped.
   obj_box_str = mat2str(obj_box{ims_i}([2 4]));
   obj_box_str = strrep(obj_box_str, ' ', '_');
   obj_box_str = obj_box_str(2:end-1);  % Strip '[' and ']'
   % Add 'x' so it's a valid field name.
   lookup.(['x' name obj_box_str]) = correct(ims_i);
end
end