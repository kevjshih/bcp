VOCyear = '2010';

% dataset superdirectory
DATADIR = '/home/kevin/data/';




VOCYEAR = ['VOC' VOCyear]; % I think this is used by exemplar SVM
if(~isdeployed)
   ROOTDIR = fileparts(which('load_init_data.m'));
else
   if(~exist('basedir', 'var'))
      get_cluster_basedir;
   end

   ROOTDIR = basedir;
end

   
WORKDIR = [ROOTDIR '/data/tmp'];
%WORKDIR = '/scratch/iendres2/tmp/';  % work is cached in this directory
PASCALDIR = fullfile(DATADIR, VOCYEAR, 'VOCdevkit/'); % Don't use tildes here
FIGDIR = [ROOTDIR '/figures'];



region_dir = fullfile(DATADIR, 'proposals', VOCYEAR);
%region_dir = '/mnt/data/features/pascal2010/processed/ranked_data/';
%region_dir = fullfile(ROOTDIR, '../proposals_journal/data/proposals', ['VOC' VOCyear]);
label_dir = fullfile(WORKDIR, 'labels'); % This isn't a problem since each image has a unique filename across all pascal (except maybe 2006/2007?)
im_dir = fullfile(PASCALDIR, VOCYEAR, 'JPEGImages');
feat_dir = fullfile(WORKDIR, 'features');

dirs.region_dir = region_dir;
dirs.label_dir = label_dir;
dirs.im_dir = im_dir;
dirs.feat_dir = feat_dir;

TRAINSET = 'train';

if(0) % Uncomment this if you change any directories
if(~exist(WORKDIR, 'file'))
   mkdir(WORKDIR)
end

if(~exist(label_dir, 'file'))
   mkdir(label_dir)
end

if(~exist(feat_dir, 'file'))
   mkdir(feat_dir)
end
end
%VOCdevkit = ['~/prog/VOC' VOCyear '/VOCdevkit/'];
VOCdevkit = PASCALDIR;
