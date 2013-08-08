function models = load_candidate_models_subset(cls, user_picked, ...
                                            object_level, trial, numpos)
% returns random, initial part models
   
BDglobals;
VOCinit;

if(~exist('user_picked', 'var') || isempty(user_picked))
   user_picked = 0;
end

if(~exist('object_level', 'var'))
   object_level = 0;
end

if(object_level==1)
   user_picked = 0;
end
VOCopts.localdir = fullfile(WORKDIR, 'subset_experiments', cls, ...
                             'exemplars', num2str(numpos), num2str(trial));

%VOCopts.localdir = fullfile(WORKDIR, 'exemplars');

if(user_picked)
   basedir = sprintf('%s/user_exemplars/%s', VOCopts.localdir, cls);
elseif(object_level)
   basedir = sprintf('%s/object_exemplars/%s', VOCopts.localdir, cls);
else
   basedir = sprintf('%s/auto_exemplars/%s', VOCopts.localdir, cls);
end
data = dir(fullfile(basedir, 'exemplar*.mat'));

% Check which ones have already been computed
candidate_dir = fullfile(WORKDIR, 'subset_experiments', cls, ...
                         'candidates', num2str(numpos), num2str(trial), ...
                         cls);

%candidate_dir = fullfile(WORKDIR, 'candidates', cls);

start = tic;

skipped = zeros(1,length(data));

for i = 1:length(data)
   if(toc(start)>5)
      fprintf('%d/%d\n', i, length(data));
      start = tic;
   end
%   if(exist(fullfile(candidate_dir, data(i).name)))
     mtmp = load(fullfile(basedir, data(i).name));
     models{i} = mtmp.m.model;
     models{i}.name = data(i).name;

      models{i} = convert_part_model(convert_part_model(models{i})); % Remove extra fields

   if(object_level)
      models{i}.spat_const = [0 1 0 1 0.8 1];
   end
%   else
%      skipped(i) = 1;
%   end
end

models = models(~skipped);
