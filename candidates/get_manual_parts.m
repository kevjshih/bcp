function parts = get_manual_parts(VOCopts, set, cls)
basedir = fullfile(VOCopts.localdir, 'manual_parts');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [set '_' cls '.mat']);
if fileexists(cached_filename)
   load(cached_filename);
else
   parts = {};
end
end