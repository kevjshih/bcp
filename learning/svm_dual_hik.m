function [model alphas] = svm_dual_hik(Y, X, C, Nbins, alphas)
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

if(Nbins>256)
%   error('This method only supports <=256 bins (Sorry I was lazy!)\n');
end

Nex = size(X,2);

if(numel(Y)~=Nex)
   error('Number of labels doesn''t match number of training vectors\n');
end

% Quantize data
max_val = max(X, [], 2);
scaling = Nbins./(max_val+eps); % Compute bin index on the fly

quantX = zeros(size(X), 'uint16');

for i = 1:size(X,1)
   quantX(i, :) = max(0,min(floor(X(i,:)*scaling(i)), Nbins-1)); % 0 indexed
end


model.scaling = scaling; % This will be needed for classification
model.Nbins = Nbins;

if(exist('alphas', 'var'))
   [model.Ml model.Mu model.bias alphas] = svm_dual_hik_mex(Y, X, quantX, Nbins, C, alphas);
else
   [model.Ml model.Mu model.bias alphas] = svm_dual_hik_mex(Y, X, quantX, Nbins, C);
end
%model.Ml = Maccum; % accumulated A values
% Reconstruct model...
% cumsum(.....)

return
keyboard
% Recontruct model
Ml = zeros(size(model.Ml));
Mu = zeros(size(model.Mu));
for i = 1:size(X,1)
   Ml(i,:) = accumarray(quantX(i,:)'+1, Y.*alphas.*X(i,:)', [Nbins 1])';
   Mu(i,:) = accumarray(quantX(i,:)'+1, Y.*alphas, [Nbins 1])';
end

Mlf = cumsum(Ml(:,end:-1:1), 2);
Mlf = Mlf(:,end:-1:1);
%Mlf = bsxfun(@minus, Mlf(:,end), Mlf);
Muf = cumsum(Mu, 2);
