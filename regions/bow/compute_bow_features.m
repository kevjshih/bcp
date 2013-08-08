function [feat] = compute_features(ann, regions0, codebook)

BDglobals;

[dk bn] = fileparts(ann.filename);

bow_dir = fullfile(dirs.feat_dir, 'bow');
if(~exist(bow_dir, 'file'))
   mkdir(bow_dir);
end

filename = fullfile(bow_dir, [bn '_codemap.mat']);

if(~exist(filename, 'file'))
   im = imread(fullfile(dirs.im_dir, ann.filename));
   imsz = size(im);
   [codemap]= get_codewords(im, codebook);
   save(filename, 'codemap', 'imsz');
else
   load(filename, 'codemap', 'imsz');
end

% Center codemap on image 
difference = [imsz(2) imsz(1)] - [size(codemap,2), size(codemap,1)];
offsets = repmat(difference/2, 1, 2);

regions = max(1,round(bsxfun(@minus, regions0, offsets)));
regions = bsxfun(@min, regions, repmat([size(codemap,2), size(codemap,1)], 1, 2));


L = 3;
K = size(codebook,2);


numfeat = sum(2.^(2*([1:L]-1)))*K;

feat = zeros(numfeat, size(regions0, 1));

for i = 1:size(regions,1)
   reg = regions(i,:);

   % Compute histogram first
   level = L;

   max_bins = 2^(level-1);
   xbins = round(linspace(reg(1), reg(3)+1, max_bins+1));
   ybins = round(linspace(reg(2), reg(4)+1, max_bins+1));
   
   h = zeros(max_bins, max_bins, K);

   for x = 1:max_bins
      xmap = codemap(:, xbins(x):xbins(x+1)-1);

      for y = 1:max_bins
         map = xmap(ybins(y):ybins(y+1)-1,:);
   
         h(x,y,:) = reshape(hist(map(:), 1:K), [1 1 K]);
      end
   end

   feat0{level} = reshape(bsxfun(@rdivide, h+eps, sum(h+eps,3)) * 1/2^(L-level+1), [], 1);
   
   % Sum over histograms
   for level = 1:L-1
      new_max_bins = 2^(level-1);
      new_h = zeros(new_max_bins, new_max_bins, K);

      for x0 = 1:max_bins
         x = ceil(x0/2^(L-level));

         for y0 = 1:max_bins
            y = ceil(y0/2^(L-level));
            new_h(x,y,:) = new_h(x,y,:) + h(x0, y0, :); 
         end
      end
      feat0{level} = reshape(bsxfun(@rdivide, new_h+eps, sum(new_h+eps,3)) * 1/2^(L-level+1), [], 1);
   end

   feat(:, i) = cat(1, feat0{:});
end
