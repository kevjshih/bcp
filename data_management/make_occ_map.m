BDglobals;

addpath(genpath('~/prog/proposals/src/iccv07Final'));

d = dir(fullfile(feat_dir, 'occlusion', '*_occlusion.mat'));
mkdir(fullfile(feat_dir, 'occmap'));

for i = 1:length(d)
   fname = strrep(d(i).name, '_occlusion.mat', '_occmap.png');
   fpath = fullfile(feat_dir, 'occmap', fname);

   if(~exist(fpath, 'file'))
      dat = load(fullfile(feat_dir, 'occlusion', d(i).name));

      bmaps = getOcclusionMaps(dat.bndinfo_all);
      imwrite(fname, repmat(mean(bmaps,3), [1 1 3]));
   end
end
