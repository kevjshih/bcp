function feat = get_bound_feat(model, filename, dirs)

[dk bn] = fileparts(filename);
fname = fullfile(dirs.feat_dir, 'occ_feat', [bn '_occfeat.mat']);

if(~exist(fname, 'file'))
   im = imread(fullfile(dirs.im_dir, filename));


   load(fullfile(dirs.feat_dir, 'occlusion', [bn '_occlusion.mat']), 'bndinfo_all');

   [feat.feat feat.scales] = occ_featpyramid(im, bndinfo_all, model.sbin, model.interval);
   
   mkdir_quiet(fullfile(dirs.feat_dir, 'occ_feat'));
   save(fname, 'feat');
else
   load(fname);
end

padx = ceil(model.maxsize(2)/2+1);
pady = ceil(model.maxsize(1)/2+1);

for s = 1:numel(feat.feat) % pad it
   feat.feat{s} = padarray(feat.feat{s}, [pady-1 padx-1 0], 0, 'pre'); % Left hand side has 1 more column that HOG features from featpyramid due to a different resampling method
   feat.feat{s} = double(padarray(feat.feat{s}, [pady padx 0], 0, 'post'));
end


feat.imsizes = []; % This isn't used here
