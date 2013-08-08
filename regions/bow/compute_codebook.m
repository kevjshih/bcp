function codebook = create_codebook(D, K)

BDglobals;

cb_file = fullfile(WORKDIR, 'sift_cw.mat');

if(~exist(cb_file, 'file'))
   ann = [D.annotation];
   fnames = {ann.filename};

   images = strcat([im_dir '/'], fnames);
   
   codebook = create_sift_codebook(images, K);

   save(cb_file, 'codebook');
else
   load(cb_file, 'codebook');
end
