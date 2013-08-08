function [feat] = compute_features(im, regions0, codebook)

tic;
[codemap]= get_codewords(im, codebook);

% Center codemap on image 
difference = [size(im,2) size(im,1)] - [size(codemap,2), size(codemap,1)];
offsets = repmat(difference/2, 1, 2);

regions = max(1,round(bsxfun(@minus, regions0, offsets)));
regions = bsxfun(@min, regions, repmat([size(codemap,2), size(codemap,1)], 1, 2));


L = 3;
K = size(codebook,2);


numfeat = sum(2.^(2*([1:L]-1)))*K;

feat = zeros(numfeat, size(regions0, 1));
toc;

tic;
for i = 1:size(regions,1)
   reg = regions(i,:);

   % Lazy approach

   for level = 1:L
      max_bins = 2^(level-1);
      xbins = round(linspace(reg(1), reg(3)+1, max_bins+1));
      ybins = round(linspace(reg(2), reg(4)+1, max_bins+1));
   
      h = zeros(max_bins, max_bins, K)+eps;

      for x = 1:max_bins
         xmap = codemap(:, xbins(x):xbins(x+1)-1);

         for y = 1:max_bins
            map = xmap(ybins(y):ybins(y+1)-1,:);
   
            h(x,y,:) = reshape(hist(map(:), 1:K), [1 1 K]);
         end
      end

      feat0{level} = reshape(bsxfun(@rdivide, h+eps, sum(h+eps,3)) * 1/2^(L-level+1), [], 1);
   end

   feat(:, i) = cat(1, feat0{:});
end
toc;
