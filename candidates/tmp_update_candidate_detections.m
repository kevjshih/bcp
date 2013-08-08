function results = test_candidate_detections(cls)
   
BDglobals;
VOCinit;

basedir = sprintf('%s/user exemplars/%s', VOCopts.localdir, cls);
data = dir(fullfile(basedir, 'exemplar*.mat'));

candidate_dir = fullfile(WORKDIR, 'candidates', cls);
for i = 1:length(data)
   fprintf('%d/%d\n', i, length(data));
   mtmp = load(fullfile(basedir, data(i).name));
   models{i} = mtmp.m.model;
   models{i}.name = data(i).name;

   if(exist(fullfile(candidate_dir, models{i}.name)))
      todo(i) = 0;
      tic;      t = load(fullfile(candidate_dir, models{i}.name), 'part_scores', 'part_detections');toc;
      if(isfield(t,'part_detections'))
         part_detections = t.part_detections;
         %save(fullfile(candidate_dir, ['boxes_' models{i}.name]), 'part_detections');
         clear part_detections;
      end
      part_scores = t.part_scores;
      save(fullfile(candidate_dir, models{i}.name), '-v6', 'part_scores');
      %
      try,   clear part_scores t, end
   end
end

