function part_bboxes_struct = manual_set_part_bboxes(VOCopts, part, set, use_cached)
% Returns a struct encoding all the instances of a part in its class for
% the specified set. Set 'use_cached' to true if you don't want to
% verify the cached struct.
%
% Use 'part_bboxes_struct' for lookups with 'manual_check_part_bboxes_overlap'.

if ~exist('use_cached', 'var')
   use_cached = false;
end

basedir = fullfile(VOCopts.localdir, 'part_bboxes');
if ~exist(basedir, 'dir')
   mkdir(basedir);
end

[~, name] = fileparts(part.im);
cached_filename = fullfile(basedir, [set '_' part.class '_' name '_' mat2str(part.bbox) '.mat']);

if fileexists(cached_filename)
   fprintf(['Loading part bboxes from "' cached_filename '"...\n']);
   load(cached_filename, 'part_bboxes_struct');
   if use_cached
      return;
   end
else
   [ims obj_box] = get_object_boxes(VOCopts, set, part.cls);
   part_bboxes_struct = struct();
end

% applicability.correct = ui_check_part(part, applicability.ims, applicability.obj_box, applicability.correct);
% applicability.lookup = create_applicability_lookup(applicability.ims, applicability.obj_box, applicability.correct);
save(cached_filename, 'part_bboxes_struct');
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