function y = sigmoid(x, beta)

y = 1./(1+exp(-beta(1)*x - beta(2)));

if(length(beta)>=3)
    y = y + beta(3);
end
