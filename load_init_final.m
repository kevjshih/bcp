if(~exist('cls', 'var'))
   error('You need to define cls!');
end
% This version trains everything on train set
%basedir = fileparts(which('load_init_data.m'));
%javaaddpath(fullfile(basedir, 'external/JavaBoost/dist/JBoost.jar'));

BDglobals;
BDpascal_init;

init_dir = fullfile(basedir, 'data', 'initdata');
init_data = fullfile(init_dir, [cls, '_', VOCyear, '_init_trainval.mat']);


if(exist(init_data, 'file'))
   start1 = tic;
   load(init_data);
   stop1 = toc(start1);
   start2 = tic;
   %addpath(whole_path);
   fprintf('Loaded initial %s data: %d, updated path: %d\n', cls, stop1, toc(start2));

   if(0 && ~any(strcmp(model.region_model, 'Area model')))
      % Adding two models for area and aspect ratio
      model.region_model{end+1} = 'Area model';
      model.region_model{end+1} = 'Area model';
      
      cached_scores = add_region_area(model, D, cached_scores);
      
      save(init_data);
   end
else
   if(~exist(init_dir, 'file'))
      mkdir(init_dir);
   end

   D = pasc2D('trainval', VOCopts);
%   Dtest = pasc2D('test', VOCopts);
   extract_all_region_features(D);

   model = init_model(cls);
   cached_scores = init_cached_scores(model, D);

   model = trainval_region_model(D, cached_scores, model);
   cached_scores = add_region_scores(model, D, cached_scores);

   %model.region_model{end+1} = 'Area model';
   %model.region_model{end+1} = 'Area model';   
   %cached_scores = add_region_area(model, D, cached_scores);

   whole_path = path;
   save(init_data);
end
