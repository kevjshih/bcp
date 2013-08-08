function add_manual_parts(VOCopts, set, cls)
basedir = fullfile(VOCopts.localdir, 'manual_parts');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [set '_' cls '.mat']);

parts = get_manual_parts(VOCopts, set, cls);

new_parts = ui_pascal_choose_parts(VOCopts, set, cls);
for i = 1:length(new_parts)
   parts{end+1} = new_parts{i};
end

save(cached_filename, 'parts');
end