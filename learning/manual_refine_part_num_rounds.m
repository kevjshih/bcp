function num_rounds = manual_refine_part_num_rounds(VOCopts, part)
basedir = fullfile(VOCopts.localdir, 'manual_refinement_rounds');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [part.name '_num_rounds.mat']);
if ~fileexists(cached_filename)
   num_rounds = -1;
else
   load(cached_filename);
   num_rounds = round_i - 1;
end
end