function beta = learn_obj_prob(roc, num_param)

if(~exist('num_param', 'var'))
    num_param = 3;
end

counts = [roc.tp(1)>0; roc.tp(2:end)-roc.tp(1:end-1)];

bins = linspace(min(roc.conf(~isinf(roc.conf))), max(roc.conf(~isinf(roc.conf))), ceil(length(roc.conf)/50));

pos_hist = histc(roc.conf(counts>=1), bins);
neg_hist = histc(roc.conf(counts==0), bins);


pos_prob = pos_hist./(pos_hist+neg_hist);

% Smooth it
%pos_prob= cummax(pos_prob);

ok = (pos_hist+neg_hist) > 0; % Make sure we have some data


sig = @(p, X) sigmoid(X, p); % nlinfit is stupid and reverses the parameters

beta0 = [1e-4 0 0];
beta0 = beta0(1:num_param);

beta = nlinfit(reshape(bins(ok), [], 1), reshape(pos_prob(ok),[],1), sig, beta0); % Start with a very smooth sigmoid

clf;
bar(bins, pos_prob);
hold on;

confs = linspace(bins(1), bins(end), 10000);
plot(confs, sig(beta, confs));
drawnow;

