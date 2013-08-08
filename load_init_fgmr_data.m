if(~exist('cls', 'var'))
   error('You need to define cls!');
end
% This version trains everything on train set
basedir = fileparts(which('load_init_data.m'));
javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');
addpath('~/prog/tools/JavaBoost/');
addpath(genpath('~/prog/attributes/CORE/evaluation'))
addpath(genpath('~/prog/attributes/CORE/tools'))
addpath(genpath('~/prog/proposals'))
addpath(('~/prog/tools'))
addpath('~/prog/voc-release3.1')
addpath('~/prog/labelme')
addpath(genpath(basedir))
rmpath(genpath(fullfile(basedir, 'old')))

BDglobals;
BDpascal_init;

init_dir = fullfile(basedir, 'data', 'initdata');
init_data = fullfile(init_dir, [cls, '_', VOCyear, '_init.mat']);

if(exist(init_data, 'file'))
   start1 = tic;
   load(init_data);
   stop1 = toc(start1);
   start2 = tic;
   %addpath(whole_path);
   fprintf('Loaded initial %s data: %d, updated path: %d\n', cls, stop1, toc(start2));
else
   if(~exist(init_dir, 'file'))
      mkdir(init_dir);
   end

   D = pasc2D('train', VOCopts);
   Dtest = pasc2D('val', VOCopts);

   % Load pretrained root model
   fgmr_model = load(fullfile('data/fgmr_pretrained_v4', [cls, '_model_', VOCyear, '_train_fullwindow.mat']));

   cached_scores = init_cached_scores_fgmr(model, fgmr_model.model, D);
   cached_scores_test = init_cached_scores_fgmr(model, fgmr_model.model, Dtest);
   model = init_model(cls);

   whole_path = path;
   save(init_data);
end

