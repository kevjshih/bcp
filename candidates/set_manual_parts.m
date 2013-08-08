function set_manual_parts(VOCopts, set, cls)
basedir = fullfile(VOCopts.localdir, 'manual_parts');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [set '_' cls '.mat']);

old_parts = get_manual_parts(VOCopts, set, cls);
parts = ui_pascal_choose_parts(VOCopts, set, cls, false, [], old_parts);

if (isempty(parts))
   parts = old_parts;
end

save(cached_filename, 'parts');
end