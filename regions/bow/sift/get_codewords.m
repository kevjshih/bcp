function [codemap best_dist dists]= get_codewords(im, centers, pca_data)

if(~iscell(im) && size(im,3)<=3)
    if(size(im,3)<3)
        im = repmat(im,[1 1 3]);
    end
    [descr] = dense_sift(im);%, ceil(rand*10)+10);
else
    descr = im;
end

if(iscell(descr))
    descr = descr{1};
end
sz = size(descr);

descr = reshape(double(descr), [], sz(3))';

if(exist('pca_data', 'var') && ~isempty(pca_data))
   descr = pca_project(descr, pca_data);
end
% Compute distance to centers, find best codeword
%dists = slmetric_pw(centers, descr, 'cityblk')';
if(nargout>1)
dists = slmetric_pw(centers, descr, 'eucdist')';

[bd best] = min(dists,[],2);
else
dists= bsxfun(@minus, sum(centers.^2,1)',  2*centers'*descr);
[dk best] = min(dists,[],1);    
end
% Reshape
codemap = reshape(best, sz(1:2));
if(nargout>1)
best_dist = reshape(bd, sz(1:2));

dists = reshape(dists, [sz(1:2) size(centers,2)]);

if(exist('pca_data', 'var') && ~isempty(pca_data) && isfield(pca_data, 'sift_mean'))
    dists = bsxfun(@minus, dists, shiftdim(pca_data.sift_mean(:), -2));
    dists = bsxfun(@rdivide, dists, shiftdim(pca_data.sift_var(:), -2));
end
end
