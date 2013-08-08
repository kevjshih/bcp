function refinement_round = manual_refine_part_round(VOCopts, part, round)
% retrieves the data for the specified round of refinement for the part
% TODO: modify so that each round of refinement is in its own file (cuts down on memory usage)

basedir = fullfile(VOCopts.localdir, 'manual_refinement_rounds');
if ~exist(basedir, 'dir');
   mkdir(basedir);
end

cached_filename = fullfile(basedir, [part.name ' round ' num2str(round) '.mat']);

if ~fileexists(cached_filename)
   disp(['Error: part not refined ' num2str(round) ' times yet']);
else
   fprintf(['Loading refinement round from "' cached_filename '"...\n']);
   load(cached_filename);
end
end