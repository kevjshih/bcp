function y = sigmoid(x, beta)
% assumes 
y = 1./(1+exp(bsxfun(@minus, bsxfun(@times, -beta(1, :), x),  beta(2,:))));

if(size(beta,1)>=3)
    y = bsxfun(@plus, y, beta(3, :));
end
