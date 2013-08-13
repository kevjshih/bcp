if(~exist('cls', 'var'))
   error('You need to define cls!');
end

basedir = fileparts(which('load_init_data.m'));
javaaddpath(fullfile(basedir, 'external/JavaBoost/Java/dist/JBoost.jar'));

BDglobals;
BDpascal_init;

init_dir = fullfile(basedir, 'data', 'initdata');
init_data = fullfile(init_dir, [cls, '_', VOCyear, '_init_test.mat']);


if(exist(init_data, 'file'))
   start1 = tic;
   load(init_data);
   stop1 = toc(start1);
   start2 = tic;
   %addpath(whole_path);
   fprintf('Loaded initial %s data: %d, updated path: %d\n', cls, stop1, toc(start2));
else
    
%if(~exist('model', 'var')) % This is where things get sloppy, ideally should keep track of train year and test year
    load(fullfile(basedir, 'data', 'initdata', [cls, '_', '2010', '_init_trainval.mat']), 'model');
%end
    
   if(~exist(init_dir, 'file'))
      mkdir(init_dir);
   end

   Dtest = pasc2D('test', VOCopts);
   extract_all_region_features(Dtest);
   cached_scores_test = init_cached_scores(model, Dtest);
   cached_scores_test = add_region_scores(model, Dtest, cached_scores_test);
%   cached_scores = add_region_area(model, D, cached_scores);

   whole_path = path;
   save(init_data, 'cached_scores_test', 'Dtest');
end
