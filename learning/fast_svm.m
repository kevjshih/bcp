function [w b alpha]= fast_svm(Y, X, C, wpos)
% Fast L2 regularized SVM
% 
% [w b] = fast_svm(Y, X, C)
%
% Input:
%   Y: vector in {-1, 1} indicating negative or positive respectively
%   X: Nfeat x Nex, one column for each training example
%   C: Regularization parameter
%
% Output:
%   w: 1 x Nex, model weight vector
%   b: bias
%
% Classification is w*X + b


if(~isdouble(Y))
   Y = double(Y);
end

if(~isdouble(X))
   X = double(X);
end

if(~isdouble(C))
   C = double(C);
end

weighting = ones(size(Y));

if(exist('wpos', 'var'))
   weighting(Y==1) = wpos;
end

[w0 alpha] = svm_weighted_dual_mex(Y, X, weighting, C);

w = w0(1:end-1);
b = w0(end);


function b = isdouble(X)

b = isa(X, 'double');
