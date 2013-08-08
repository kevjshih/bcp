function beta = learn_obj_prob2(roc, num_param)

if(~exist('num_param', 'var'))
    num_param = 3;
end

% Smooth roc curve to be monotonic
roc.p(end:-1:1) = cummax(roc.p(end:-1:1));

% Delete infinities
infs = isinf(roc.conf);
roc.p(infs) = [];
roc.r(infs) = [];
roc.conf(infs) = [];

beta0 = [eps -1 0];
beta = fminunc(@(x)compute_obj(x, roc), beta0(1:num_param));




function [f g] = compute_obj(beta, roc)

pred = sigmoid(roc.conf, beta);

cum_pred = cumsum(pred);

weights = (roc.r(2:end) - roc.r(1:end-1));
cum_weights = cumsum(weights);

counts = (1:length(roc.r))';

f = weights'*((roc.p(1:end-1) - cum_pred(1:end-1)./counts(1:end-1)).^2);
