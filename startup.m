fprintf('Executing Startup!\n');
st = tic;
javaaddpath('external/JavaBoost/Java/dist/JBoost.jar');
basedir = fileparts(which('load_init_data.m'));


addpath(genpath(fullfile(basedir, 'external', 'CORE')));
addpath(genpath(fullfile(basedir, 'external', ...
                         'cvpr2010_repredict')));
addpath(genpath(fullfile(basedir, 'external', ['fast-additive-' ...
                    'svms'])));
addpath(genpath(fullfile(basedir, 'external', 'labelme')));
addpath(genpath(fullfile(basedir, 'external', 'minFunc')));
addpath(genpath(fullfile(basedir, 'external', 'minConf')));
addpath(genpath(fullfile(basedir, 'external', 'whog')));
addpath(genpath(fullfile(basedir, 'external', 'proposals')));
addpath(genpath(fullfile(basedir, 'external', 'textons')));
addpath(fullfile(basedir, 'external'));
addpath(fullfile(basedir, 'analysis'));

addpath(genpath(fullfile(basedir, 'candidates')));
addpath(fullfile(basedir, 'data_management'));
addpath(fullfile(basedir, 'inference'));
addpath(fullfile(basedir, 'init'));
addpath(fullfile(basedir, 'learning'));
addpath(genpath(fullfile(basedir, 'regions')));
addpath(fullfile(basedir, 'soft_nms'));
addpath(fullfile(basedir, 'test'));
addpath(fullfile(basedir, 'cluster'));
addpath(fullfile(basedir, 'class_confusion'));

BDglobals;
BDpascal_init;

addpath(PASCALDIR);
addpath(fullfile(PASCALDIR, 'VOCcode'));
fprintf('Finished in %d seconds\n', ceil(toc(st)));

