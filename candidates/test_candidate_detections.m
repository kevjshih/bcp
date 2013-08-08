function results_out = test_candidate_detections(D, cached_scores, cls, models, N, N2, suffix, dirtodo)


im_dir = [];
BDglobals;
VOCinit;

if(~exist('suffix','var'))
    suffix = '';
end

if(~isempty(suffix))
   suffix = ['_' suffix];
end

VOCopts.localdir = fullfile(WORKDIR, ['exemplars' suffix]);

if(~exist('dirtodo','var'))
   dirtodo = 'auto_exemplars';
end

if(isempty(models)) % Load a random sample
   clear models;
   basedir = sprintf('%s/%s/%s', VOCopts.localdir, dirtodo, cls);
   data = dir(fullfile(basedir, 'exemplar*.mat'));

%   r = 1:length(data);
   if(~exist('N2', 'var')) % Randomly sample
      r = randperm(length(data));
      data = data(r(1:min(N,end)));
   else % Do a block
      block_id = N2;
      num_blocks = N;
      blocks = round(linspace(1,numel(data)+1, N+1));
      block = blocks(block_id):blocks(block_id+1)-1;
      fprintf('Block %d...Running from %d to %d\n', block_id, blocks(block_id),blocks(block_id+1)-1);
      data = data(block);
   end

   for i = 1:length(data)
      mtmp = load(fullfile(basedir, data(i).name));
      models{i} = mtmp.m.model;
      models{i}.name = data(i).name;
   end
end

% Check which ones have already been computed
candidate_dir = fullfile(WORKDIR, 'candidates', cls);
if(~exist(candidate_dir, 'file'))
   mkdir(candidate_dir);
end

results = cell(length(models), 1);
%if(strcmp(cls,'car'));
%   todo = ones(length(models), 1);
%else
if(nargout>0)
   todo = zeros(length(models), 1); % Don't want to compute anything
else
   todo = ones(length(models), 1); % Only calling it to do computation...
end
%end

if(~iscell(models))
    models = {models};
end

if(nargout>0)
for i = 1:length(models)
%   if(exist(fullfile(candidate_dir, models{i}.name)))
    try
      restmp = load(fullfile(candidate_dir, models{i}.name), 'part_scores');
      results{i} = restmp.part_scores;
      todo(i) = 0;
   end
end
else
   fprintf('Running everything\n');
end

todo = find(todo);
all_models = models(todo);


regions = cell(1, length(D));
ids = cell(1, length(D));
scores= cell(1, length(D));
parts= cell(1, length(D));

if(~isempty(todo))
    parfor i = 1:min(length(D),5000) % Don't need to test on every image
      fprintf('%d\n', i);
      %      try
         im = imread(fullfile(im_dir, D(i).annotation.filename));
         regions{i} = cached_scores{i}.regions;
         [dk ids{i}] = fileparts(D(i).annotation.filename);
         [scores{i} parts{i}] = part_inference(im, all_models, regions{i});
         %catch
         %fprintf('Error on %s\n', D(i).annotation.filename);
         %end
   end

   all_scores = cell(length(parts), length(all_models));
   all_parts = cell(length(parts), length(all_models));
   for i = 1:length(parts)
       if(~isempty(scores{i}))
            all_scores(i,:) = scores{i};
            all_parts(i,:) = parts{i};
       end 
   end

   for i = 1:length(todo)
      score_file = fullfile(candidate_dir, all_models{i}.name);
      part_file = fullfile(candidate_dir, ['locs_' all_models{i}.name]);
      todo_ind = todo(i);
   
      part_scores = all_scores(:, i);
      part_detections = all_parts(:, i);
     if(nargout>0) 
      results{todo_ind} = part_scores;
      end
      save(score_file, 'part_scores');
      %save(score_file, 'part_scores', '-v6'); % to speed up loading
      %save(part_file, 'part_detections'); % allow compression, this one gets big!
   end
end

if(nargout>0)
   results_out = results;
end
