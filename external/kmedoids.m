function [label, energy, index] = kmedoids(X,k, init)
% [label, energy, index] = kmedoids(X,k, init)
% X: d x n data matrix
% k: number of cluster
% Written by Mo Chen (sth4nth@gamil.com)
v = dot(X,X,1);
D = bsxfun(@plus,v,v')-2*(X'*X);
n = size(X,2);

if(~exist('init','var'))
    [dc, label] = min(D(randsample(n,k),:));
else
    [dc, label] = min(D(init, :));
end

last = 0;
while any(label ~= last)
    [~, index] = min(D*sparse(1:n,label,1,n,k,n));
    last = label;
    [val, label] = min(D(index,:),[],1);
end
energy = sum(val);
