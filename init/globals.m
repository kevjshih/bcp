warning('This function is obsolete!, please change to BDglobals.m');

VOCYEAR = 'VOC2010';
VOCyear = '2010';
WORKDIR = '/home/engr/iendres2/prog/boosted_detection/data/tmp';
%WORKDIR = '/scratch/iendres2/tmp/';  % work is cached in this directory
PASCALDIR = '/home/engr/iendres2/prog/VOC2010/VOCdevkit/'; % Don't use tildes here

%region_dir = '/mnt/data/features/pascal2010/processed/ranked_data/';
region_dir = '/home/engr/iendres2/prog/proposals_journal/data/proposals/VOC2010';
label_dir = fullfile(WORKDIR, 'labels');
im_dir = fullfile(PASCALDIR, 'VOC2010/JPEGImages');
feat_dir = fullfile(WORKDIR, 'features');

dirs.region_dir = region_dir;
dirs.label_dir = label_dir;
dirs.im_dir = im_dir;
dirs.feat_dir = feat_dir;

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
VOCdevkit = ['~/prog/VOC' VOCyear '/VOCdevkit/'];
