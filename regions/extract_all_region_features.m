function extract_all_region_features(D)

BDglobals;

region_feat_dir = fullfile(dirs.feat_dir, 'region');
mkdir(region_feat_dir);

dirs.precomputed_feat = fullfile(dirs.feat_dir, 'tc2');

parfor i = 1:length(D)
    try
      fprintf('%d/%d\n', i, length(D));
      extract_region_features(D(i).annotation, dirs);
    catch
        disp('region failed to extract');
    end
end
